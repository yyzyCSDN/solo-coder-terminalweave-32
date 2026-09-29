from datetime import datetime, timedelta, timezone

from terminalweave.transshipment.connections import Connection, ConnectionGraph, ServiceCall
from terminalweave.transshipment.rebooking import Sailing
from terminalweave.transshipment.recovery import (
    ContainerStage,
    ConnectionRecoveryPlanner,
    RecoveryStatus,
    ScheduleChange,
)
from terminalweave.transshipment.dwell import DwellRecord
from terminalweave.yard.inventory import InventoryEvent, InventoryLedger


UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)


def call(call_id, service, arrival, departure, destination=None):
    return ServiceCall(call_id, service, arrival, departure, destination)


def sailing(sailing_id, service, destination, departure, capacity, booked=0, call_id=None):
    return Sailing(sailing_id, service, destination, departure, capacity, booked, call_id)


def graph_with(calls, connections):
    graph = ConnectionGraph()
    for service_call in calls:
        graph.add_call(service_call)
    for connection in connections:
        graph.add_connection(connection)
    return graph


def delayed_inbound_graph():
    inbound = call('IN', 'S1', D, D + timedelta(hours=2), 'ORIGIN')
    outbound = call('OUT', 'S2', D + timedelta(hours=5), D + timedelta(hours=8), 'DEST')
    revised_inbound = call('IN', 'S1', D + timedelta(hours=7), D + timedelta(hours=9), 'ORIGIN')
    connection = Connection('C1', 'IN', 'OUT', timedelta(hours=2))
    return graph_with([inbound, outbound], [connection]), revised_inbound


def test_recovery_selects_achievable_preferred_service_sailing():
    graph, revised_inbound = delayed_inbound_graph()
    planner = ConnectionRecoveryPlanner(graph)
    candidates = [
        sailing('OTHER', 'S9', 'DEST', D + timedelta(hours=10), 2),
        sailing('NEXT', 'S2', 'DEST', D + timedelta(hours=12), 2, call_id='NEXT'),
        sailing('OLD', 'S2', 'DEST', D + timedelta(hours=8), 2, call_id='OUT'),
    ]

    plan = planner.plan(
        [ScheduleChange('IN', revised_inbound)],
        candidates,
        destinations={'C1': 'DEST'},
    )

    assert plan.status is RecoveryStatus.RECOVERED
    assert plan.ok
    assert plan.failure_count == 0
    assert plan.affected[0].container_id == 'C1'
    selection = plan.recovered[0]
    assert selection.container_id == 'C1'
    assert selection.sailing.sailing_id == 'NEXT'
    assert selection.outbound_call == 'NEXT'
    assert selection.new_connection.outbound_call == 'NEXT'
    assert plan.unaffected == ()


def test_recovery_reports_partial_failure_without_hiding_it():
    inbound = call('IN', 'S1', D, D + timedelta(hours=2))
    outbound = call('OUT', 'S2', D + timedelta(hours=5), D + timedelta(hours=8), 'DEST')
    revised_inbound = call('IN', 'S1', D + timedelta(hours=7), D + timedelta(hours=9))
    connections = [
        Connection('C1', 'IN', 'OUT', timedelta(hours=2)),
        Connection('C2', 'IN', 'OUT', timedelta(hours=2)),
    ]
    graph = graph_with([inbound, outbound], connections)
    candidates = [sailing('NEXT', 'S2', 'DEST', D + timedelta(hours=12), 1, call_id='NEXT')]

    plan = ConnectionRecoveryPlanner(graph).plan(
        {'IN': revised_inbound},
        candidates,
    )

    assert plan.status is RecoveryStatus.PARTIAL
    assert not plan.ok
    assert {x.container_id for x in plan.recovered} == {'C1'}
    assert [x.container_id for x in plan.failures] == ['C2']
    assert plan.failures[0].reason == 'no eligible recovery sailing'


def test_loaded_and_delivered_containers_are_immutable():
    inbound = call('IN', 'S1', D, D + timedelta(hours=2))
    outbound = call('OUT', 'S2', D + timedelta(hours=5), D + timedelta(hours=8), 'DEST')
    graph = graph_with(
        [inbound, outbound],
        [
            Connection('C1', 'IN', 'OUT', timedelta(hours=2)),
            Connection('C2', 'IN', 'OUT', timedelta(hours=2)),
        ],
    )
    planner = ConnectionRecoveryPlanner(
        graph,
        statuses={
            'C1': ContainerStage.LOADED,
            'C2': 'delivered',
        },
    )

    plan = planner.plan(
        [ScheduleChange('IN', call('IN', 'S1', D + timedelta(hours=7), D + timedelta(hours=9)))],
        [sailing('NEXT', 'S2', 'DEST', D + timedelta(hours=12), 5, call_id='NEXT')],
    )

    assert plan.status is RecoveryStatus.FAILED
    assert plan.recovered == ()
    assert [x.container_id for x in plan.failures] == ['C1', 'C2']
    assert all('already' in x.reason for x in plan.failures)


def test_inventory_ledger_loaded_event_blocks_rebooking():
    graph, revised_inbound = delayed_inbound_graph()
    ledger = InventoryLedger()
    ledger.append(InventoryEvent(1, 'C1', 'discharged', D, 'YARD-A'))
    ledger.append(InventoryEvent(2, 'C1', 'loaded', D + timedelta(hours=3), None, 'OUT'))

    plan = ConnectionRecoveryPlanner(graph, ledger=ledger).plan(
        [ScheduleChange('IN', revised_inbound)],
        [sailing('NEXT', 'S2', 'DEST', D + timedelta(hours=12), 1, call_id='NEXT')],
        destinations={'C1': 'DEST'},
    )

    assert plan.status is RecoveryStatus.FAILED
    assert plan.failures[0].reason == 'container already loaded'


def test_dwell_record_with_loaded_time_blocks_rebooking():
    graph, revised_inbound = delayed_inbound_graph()
    dwell = DwellRecord(
        'C1',
        D + timedelta(hours=2),
        D + timedelta(hours=4),
        'DEST',
    )

    plan = ConnectionRecoveryPlanner(graph, dwell_records=[dwell]).plan(
        [ScheduleChange('IN', revised_inbound)],
        [sailing('NEXT', 'S2', 'DEST', D + timedelta(hours=12), 1, call_id='NEXT')],
        destinations={'C1': 'DEST'},
    )

    assert plan.status is RecoveryStatus.FAILED
    assert plan.failures[0].reason == 'container already loaded'


def test_unchanged_feasible_connections_are_not_replanned():
    graph, _ = delayed_inbound_graph()
    revised_outbound = call(
        'OUT',
        'S2',
        D + timedelta(hours=5),
        D + timedelta(hours=10),
        'DEST',
    )

    plan = ConnectionRecoveryPlanner(graph).plan(
        [ScheduleChange('OUT', revised_outbound)],
        [sailing('LATER', 'S2', 'DEST', D + timedelta(days=2), 1)],
    )

    assert plan.status is RecoveryStatus.NOT_REQUIRED
    assert plan.affected == ()
    assert len(plan.unaffected) == 1
    assert plan.recovered == ()
    assert plan.failures == ()

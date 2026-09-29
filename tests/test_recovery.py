from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.transshipment.connections import ServiceCall, Connection, ConnectionGraph
from terminalweave.transshipment.rebooking import Sailing
from terminalweave.transshipment.recovery import (
    ContainerStatus,
    ShipmentContext,
    ScheduleUpdate,
    SailingLedger,
    RecoveryPlanner,
    RecoveryIncompleteError,
    InfeasibilityReason,
    FailureReason,
)

UTC = timezone.utc
NOW = datetime(2026, 6, 1, tzinfo=UTC)


def build_graph():
    g = ConnectionGraph()
    g.add_call(ServiceCall('IN', 'FE1', NOW, NOW + timedelta(hours=8)))
    g.add_call(ServiceCall('OUT', 'FE2', NOW + timedelta(days=1),
                           NOW + timedelta(days=1, hours=4)))
    return g


def test_delayed_inbound_is_detected_and_rebooked():
    g = build_graph()
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))

    # Inbound pushed 2 days: ready after the original outbound departure.
    updates = [ScheduleUpdate('IN', NOW + timedelta(days=2),
                              NOW + timedelta(days=2, hours=8))]
    later = Sailing('OUT2', 'FE3', 'SGS',
                    NOW + timedelta(days=3), 100, booked_teu=0)
    planner = RecoveryPlanner(g, SailingLedger([later]))
    plan = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.IN_YARD, 'SGS', 2),
    ], NOW)

    assert plan.affected_ids == ['C1']
    affected = plan.affected[0]
    assert affected.reason is InfeasibilityReason.INBOUND_LATE
    assert affected.ready_at == NOW + timedelta(days=2, hours=4)
    assert plan.rebooked_ids == ['C1']
    assert plan.rebooked[0].sailing.sailing_id == 'OUT2'
    assert plan.rebooked[0].wait == timedelta(hours=20)
    assert plan.complete
    assert plan.summary() == {'affected': 1, 'rebooked': 1, 'failed': [],
                              'locked': [], 'missing_context': [], 'complete': True}


def test_early_outbound_advancing_is_flagged():
    g = build_graph()
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))

    # Outbound leaves only 2h after inbound arrival, transfer needs 4h.
    updates = [ScheduleUpdate('OUT', NOW + timedelta(hours=2),
                              NOW + timedelta(hours=3))]
    planner = RecoveryPlanner(g, SailingLedger([]))
    plan = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.EXPECTED, 'SGS', 2),
    ], NOW)

    assert plan.affected[0].reason is InfeasibilityReason.OUTBOUND_EARLY
    assert plan.failed_ids == ['C1']
    with pytest.raises(RecoveryIncompleteError):
        planner.ensure_complete(plan)


def test_loaded_and_delivered_boxes_are_locked_not_rebooked():
    g = build_graph()
    g.add_call(ServiceCall('IN2', 'FE1', NOW, NOW + timedelta(hours=8)))
    g.add_call(ServiceCall('OUT3', 'FE2', NOW + timedelta(days=1),
                           NOW + timedelta(days=1, hours=4)))
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))
    g.add_connection(Connection('C2', 'IN2', 'OUT3', timedelta(hours=4)))

    updates = [
        ScheduleUpdate('IN', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
        ScheduleUpdate('IN2', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
    ]
    later = Sailing('OUT2', 'FE3', 'SGS', NOW + timedelta(days=3), 100)
    planner = RecoveryPlanner(g, SailingLedger([later]))
    plan = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.LOADED, 'SGS', 2),
        ShipmentContext('C2', ContainerStatus.DELIVERED, 'SGS', 2),
    ], NOW)

    assert set(plan.affected_ids) == {'C1', 'C2'}
    assert plan.rebooked == []
    assert plan.locked_ids == ['C1', 'C2']
    reasons = {f.container_id: f.reason for f in plan.failures}
    assert reasons['C1'] is FailureReason.LOCKED_STATE
    # Committing a plan with locked boxes reserves nothing and is reported.
    assert planner.commit(plan) == {}
    with pytest.raises(RecoveryIncompleteError) as exc:
        planner.ensure_complete(plan)
    assert set(exc.value.locked) == {'C1', 'C2'}


def test_partial_batch_failure_lists_each_box():
    g = build_graph()
    for cid in ('C1', 'C2', 'C3'):
        inbound = cid.replace('C', 'IN')
        g.add_call(ServiceCall(inbound, 'FE1', NOW, NOW + timedelta(hours=8)))
        outbound = cid.replace('C', 'OUT')
        g.add_call(ServiceCall(outbound, 'FE2', NOW + timedelta(days=1),
                               NOW + timedelta(days=1, hours=4)))
        g.add_connection(Connection(cid, inbound, outbound, timedelta(hours=4)))

    updates = [
        ScheduleUpdate('IN1', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
        ScheduleUpdate('IN2', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
        ScheduleUpdate('IN3', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
    ]
    # Only one box fits the single later sailing.
    later = Sailing('OUTL', 'FE3', 'SGS', NOW + timedelta(days=3), capacity_teu=100,
                    booked_teu=97)
    planner = RecoveryPlanner(g, SailingLedger([later]))
    plan = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.IN_YARD, 'SGS', 2),
        ShipmentContext('C2', ContainerStatus.IN_YARD, 'SGS', 2),
        ShipmentContext('C3', ContainerStatus.IN_YARD, 'SGS', 2),
    ], NOW)

    assert set(plan.affected_ids) == {'C1', 'C2', 'C3'}
    assert plan.rebooked_ids == ['C1']
    assert plan.failed_ids == ['C2', 'C3']
    assert not plan.complete
    failures = {f.container_id for f in plan.failures}
    assert failures == {'C2', 'C3'}

    # Commit reserves only the one successful rebooking.
    booked = planner.commit(plan)
    assert booked == {'C1': 'OUTL'}
    assert planner.ledger.remaining('OUTL') == 1
    with pytest.raises(RecoveryIncompleteError) as exc:
        planner.ensure_complete(plan)
    assert exc.value.failures == ['C2', 'C3']


def test_unaffected_connections_are_ignored():
    g = build_graph()
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))
    # Small delay that the original slack absorbs.
    updates = [ScheduleUpdate('IN', NOW + timedelta(hours=1),
                              NOW + timedelta(hours=9))]
    planner = RecoveryPlanner(g, SailingLedger([]))
    plan = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.IN_YARD, 'SGS', 2),
    ], NOW)
    assert plan.affected == []
    assert plan.complete


def test_commit_rolls_back_when_ledger_runs_out():
    g = build_graph()
    g.add_call(ServiceCall('IN2', 'FE1', NOW, NOW + timedelta(hours=8)))
    g.add_call(ServiceCall('OUT3', 'FE2', NOW + timedelta(days=1),
                           NOW + timedelta(days=1, hours=4)))
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))
    g.add_connection(Connection('C2', 'IN2', 'OUT3', timedelta(hours=4)))
    updates = [
        ScheduleUpdate('IN', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
        ScheduleUpdate('IN2', NOW + timedelta(days=2),
                       NOW + timedelta(days=2, hours=8)),
    ]
    # Tentative view lets both decisions target the same sailing; an outside
    # booking before commit leaves room for only one.
    sailing = Sailing('OUTL', 'FE3', 'SGS', NOW + timedelta(days=3), capacity_teu=100)
    ledger = SailingLedger([sailing])
    planner = RecoveryPlanner(g, ledger)
    plan = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.IN_YARD, 'SGS', 2),
        ShipmentContext('C2', ContainerStatus.IN_YARD, 'SGS', 2),
    ], NOW)
    ledger.reserve('OUTL', 98)
    with pytest.raises(Exception):
        planner.commit(plan)
    # First reservation was rolled back.
    assert ledger.get('OUTL').booked_teu == 98


def test_missing_context_is_failure_not_silent_skip():
    g = build_graph()
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))
    updates = [ScheduleUpdate('IN', NOW + timedelta(days=2),
                              NOW + timedelta(days=2, hours=8))]
    planner = RecoveryPlanner(g, SailingLedger([]))
    plan = planner.plan(updates, [], NOW)
    assert plan.affected_ids == ['C1']
    assert plan.failures[0].reason is FailureReason.MISSING_CONTEXT
    assert not plan.complete


def test_plan_is_repeatable_before_commit():
    g = build_graph()
    g.add_connection(Connection('C1', 'IN', 'OUT', timedelta(hours=4)))
    updates = [ScheduleUpdate('IN', NOW + timedelta(days=2),
                              NOW + timedelta(days=2, hours=8))]
    later = Sailing('OUT2', 'FE3', 'SGS', NOW + timedelta(days=3), 100)
    ledger = SailingLedger([later])
    planner = RecoveryPlanner(g, ledger)
    first = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.IN_YARD, 'SGS', 2),
    ], NOW)
    second = planner.plan(updates, [
        ShipmentContext('C1', ContainerStatus.IN_YARD, 'SGS', 2),
    ], NOW)
    assert first.rebooked_ids == second.rebooked_ids == ['C1']
    assert ledger.remaining('OUT2') == 100


def test_unknown_call_in_update_is_rejected():
    g = build_graph()
    planner = RecoveryPlanner(g, SailingLedger([]))
    with pytest.raises(Exception):
        planner.plan([ScheduleUpdate('GHOST', NOW, NOW + timedelta(hours=1))],
                     [], NOW)

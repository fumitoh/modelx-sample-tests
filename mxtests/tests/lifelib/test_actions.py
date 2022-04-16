import modelx as mx
import lifelib
import pandas as pd


def test_actions(tmp_path):
    lifelib.create('savings', tmp_path / 'savings')
    m = mx.read_model(tmp_path / 'savings' / 'CashValue_ME')
    actions = m.generate_actions([m.Projection.result_pv.node()])
    m.Projection.model_point_table = m.Projection.model_point_10000
    expected = m.Projection.result_pv()
    m.clear_all()
    m.execute_actions(actions)
    pd.testing.assert_frame_equal(m.Projection.result_pv(), expected)






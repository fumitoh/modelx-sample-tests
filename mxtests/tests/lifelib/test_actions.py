import modelx as mx
import lifelib
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def savings_models(tmp_path_factory):
    lib_dir = tmp_path_factory.mktemp('sample') / 'savings'
    lifelib.create('savings', lib_dir)
    return lib_dir


def test_serialize_actions(savings_models):
    """Test savings actions in a model"""

    m = mx.read_model(savings_models / 'CashValue_ME')
    m.Projection.actions = m.generate_actions([m.Projection.result_pv.node()])
    expected = m.Projection.result_pv()
    m.clear_all()
    m.write(savings_models / 'CashValue_ME_with_actions')
    m.close()

    m2 = mx.read_model(savings_models / 'CashValue_ME_with_actions')
    m2.execute_actions(m2.Projection.actions)

    for c in m2.Projection.cells.values():
        if c.name == 'result_pv':
            # Check the target cells has the input value as an input
            assert c.is_input()
            pd.testing.assert_frame_equal(c(), expected)
        else:
            # Check all but the target cells are cleared
            assert not dict(c)

    m2.close()


def test_actions(savings_models):

    m = mx.read_model(savings_models / 'CashValue_ME')
    actions = m.generate_actions([m.Projection.result_pv.node()])
    m.Projection.model_point_table = m.Projection.model_point_10000
    expected = m.Projection.result_pv()
    m.clear_all()
    m.execute_actions(actions)
    pd.testing.assert_frame_equal(m.Projection.result_pv(), expected)
    m.close()






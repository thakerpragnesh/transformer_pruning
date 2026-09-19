import pytest
import torch
import torch.nn as nn

from pruning_transformer import (
    ActivationAwareSelector,
    CalibrationStats,
    FFNContext,
    InOutNormSelector,
    Max3SaliencyScorer,
    SaliencySelector,
    TwinRedundancySelector,
)


def make_ctx(neurons=10, hidden=4, stats=None):
    torch.manual_seed(0)
    intermediate = nn.Linear(hidden, neurons)
    output = nn.Linear(neurons, hidden)
    return FFNContext.from_layers(intermediate, output, stats=stats)


def test_saliency_selector_keeps_the_strongest():
    ctx = make_ctx(neurons=10)
    keep = SaliencySelector(Max3SaliencyScorer(), prune_percent=40).select_keep_indices(ctx)
    assert len(keep) == 6
    scores = Max3SaliencyScorer().score(ctx.intermediate_weight)
    assert set(keep) == set(torch.topk(scores, 6).indices.tolist())


def test_keep_count_rounds_rather_than_truncates():
    """int() truncates toward zero, so it prunes one neuron more than asked
    whenever n * (1 - p/100) isn't whole -- a one-directional bias.
    """
    assert SaliencySelector(Max3SaliencyScorer(), prune_percent=40).n_keep(768) == 461
    assert int(768 * 0.6) == 460  # what the old code kept
    assert SaliencySelector(Max3SaliencyScorer(), prune_percent=30).n_keep(1024) == 717
    assert int(1024 * 0.7) == 716
    assert SaliencySelector(Max3SaliencyScorer(), prune_percent=40).n_keep(10) == 6


def test_a_layer_is_never_pruned_out_of_existence():
    selector = SaliencySelector(Max3SaliencyScorer(), prune_percent=99.9, min_keep=1)
    assert selector.n_keep(10) == 1
    ctx = make_ctx(neurons=10)
    assert len(selector.select(ctx).keep_indices) == 1


def test_prune_percent_is_validated():
    with pytest.raises(ValueError, match="must be in"):
        SaliencySelector(Max3SaliencyScorer(), prune_percent=100)
    with pytest.raises(ValueError, match="must be in"):
        SaliencySelector(Max3SaliencyScorer(), prune_percent=-1)


def test_in_out_norm_sinks_a_neuron_with_a_dead_output_column():
    """The point of the product form: a strongly-driven neuron nothing
    reads is worthless, and a W_in-only criterion cannot see that.
    """
    hidden, neurons = 4, 6
    intermediate = nn.Linear(hidden, neurons)
    output = nn.Linear(neurons, hidden)
    with torch.no_grad():
        intermediate.weight[2] *= 50.0   # loudest neuron by any W_in measure
        output.weight[:, 2] = 0.0        # ...but nothing downstream reads it
    ctx = FFNContext.from_layers(intermediate, output)

    weight_only = SaliencySelector(Max3SaliencyScorer(), prune_percent=50).select_keep_indices(ctx)
    in_out = InOutNormSelector(prune_percent=50).select_keep_indices(ctx)

    assert 2 in weight_only
    assert 2 not in in_out


def test_activation_aware_drops_a_neuron_that_never_fires():
    ctx_stats = CalibrationStats(
        mean=torch.ones(6), mean_abs=torch.ones(6), rms=torch.ones(6), tokens=100,
    )
    ctx_stats.mean_abs[4] = 0.0  # neuron 4 is well-wired but inert on real data
    ctx = make_ctx(neurons=6, stats=ctx_stats)
    keep = ActivationAwareSelector(prune_percent=20).select_keep_indices(ctx)
    assert 4 not in keep


def test_activation_aware_explains_missing_calibration():
    ctx = make_ctx(neurons=6, stats=None)
    with pytest.raises(ValueError, match="needs calibration statistics"):
        ActivationAwareSelector(prune_percent=20).select_keep_indices(ctx)


def test_stats_for_the_wrong_layer_are_caught():
    wrong = CalibrationStats(torch.ones(3), torch.ones(3), torch.ones(3), tokens=10)
    ctx = make_ctx(neurons=6, stats=wrong)
    with pytest.raises(ValueError, match="stats for 3 neurons"):
        ActivationAwareSelector(prune_percent=20).select_keep_indices(ctx)


def test_activation_aware_statistic_is_validated():
    with pytest.raises(ValueError, match="mean_abs"):
        ActivationAwareSelector(prune_percent=10, statistic="variance")


def test_twin_selector_drops_the_higher_index_of_each_pair():
    ctx = make_ctx(neurons=8)
    selection = TwinRedundancySelector([(1, 4), (2, 6)]).select(ctx)
    assert selection.keep_indices == [0, 1, 2, 3, 5, 7]
    assert selection.merge_map is None


def test_twin_chains_resolve_to_a_single_survivor():
    """Given (1,4) and (4,6), neuron 6 must be merged into 1 -- merging it
    into 4 would point at a neuron that is itself being removed.
    """
    ctx = make_ctx(neurons=8)
    selector = TwinRedundancySelector([(1, 4), (4, 6)], merge=True)
    selection = selector.select(ctx)
    assert selection.keep_indices == [0, 1, 2, 3, 5, 7]
    assert selection.merge_map[4][0] == 1
    assert selection.merge_map[6][0] == 1


def test_twin_merge_scale_uses_calibration_when_available():
    stats = CalibrationStats(
        mean=torch.tensor([1.0, 2.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.ones(4), tokens=10,
    )
    ctx = make_ctx(neurons=4, stats=stats)
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    # Neuron 1 fires twice as hard as neuron 0, so its column scales by 2.
    assert selection.merge_map[1] == (0, 2.0)


def test_twin_merge_scale_falls_back_to_one_without_calibration():
    ctx = make_ctx(neurons=4, stats=None)
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, 1.0)


def test_twin_merge_scale_guards_a_near_zero_denominator():
    """A GELU neuron's signed mean can sit near zero while it is highly
    active; an unguarded ratio would explode the merged column.
    """
    stats = CalibrationStats(
        mean=torch.tensor([1e-12, 5.0, 1.0, 1.0]), mean_abs=torch.ones(4),
        rms=torch.ones(4), tokens=10,
    )
    ctx = make_ctx(neurons=4, stats=stats)
    selection = TwinRedundancySelector([(0, 1)], merge=True).select(ctx)
    assert selection.merge_map[1] == (0, 1.0)

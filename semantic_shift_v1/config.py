import json
from pathlib import Path

INTERFACE_VERSION = "semantic_shift_v1"
EXPECTED_CONDITIONS = ("B0","A0","B3","A1","A2","A3","A4","A5")
EXPECTED_SEEDS = (0,1,2,3,4)
VV_MEAN = -10.136466953246543
VV_STD = 5.013093346521575
VH_MEAN = -17.314963656896122
VH_STD = 6.202284981963544

_KD_EXPECTED = {
    "B0": (False, None, None),
    "A0": (True, "p_F", "uniform_valid"),
    "B3": (True, "p_F", "r = 1 - H(p_N,p_P,p_F)/ln(3)"),
    "A1": (True, "p_F", "w_W = 1 - p_N = p_P + p_F"),
    "A2": (True, "q_R = p_F * (1 - ReLU(p_P - p_F))", "uniform_valid"),
    "A3": (True, "q_R = p_F * (1 - ReLU(p_P - p_F))", "w_W = 1 - p_N = p_P + p_F"),
    "A4": (True, "q_H = p_F * 1[p_F >= p_P]", "w_W = 1 - p_N = p_P + p_F"),
    "A5": (False, None, None),
}

def _assert_common_preprocessing(pp):
    assert pp["band_order"] == ["VV","VH"]
    assert pp["raw_domain"] == "linear S1-GRD-derived sigma"
    assert pp["invalid_rule"] == "~finite OR x<=0 OR x>1e3"
    assert pp["invalid_fill_before_log"] == 0.0
    assert pp["floor"] == "float32 epsilon"
    assert pp["radiometric_transform"] == "exactly one 10*log10 after invalid-fill/floor"
    assert pp["clipping"] == "none"
    assert pp["double_log_forbidden"] is True
    n = pp["normalization"]
    assert n["VV"]["mean_db"] == VV_MEAN and n["VV"]["std_db"] == VV_STD
    assert n["VH"]["mean_db"] == VH_MEAN and n["VH"]["std_db"] == VH_STD

def _assert_optimizer_scheduler(c):
    assert c["epochs"] == 40 and c["batch_size"] == 8
    o = c["optimizer"]
    assert o == {
        "name":"Adam","lr":0.0003,"betas":[0.9,0.999],"eps":1e-08,
        "weight_decay":0.0,"amsgrad":False
    }
    s = c["scheduler"]
    assert s == {
        "name":"ReduceLROnPlateau","mode":"max","factor":0.5,"patience":3,
        "threshold":0.0001,"threshold_mode":"rel","cooldown":0,"min_lr":0.0,
        "eps":1e-08,"step_frequency":"once_per_validation_epoch"
    }
    assert c["precision"] == {"dtype":"FP32","amp":False}

def validate_run_config(c):
    assert c["schema_version"] == "semantic_shift_run_config_v1"
    assert c["status"] == "FROZEN"
    assert c["implementation_binding"]["required_interface_version"] == INTERFACE_VERSION
    assert c["training_authorized_at_config_freeze"] is False
    assert c["overall_g4_status_at_config_freeze"] == "OPEN"
    _assert_common_preprocessing(c["preprocessing"])
    _assert_optimizer_scheduler(c)
    ap = c["artifact_policy"]
    assert ap["drive_authoritative"] is True
    assert ap["never_overwrite_existing_run_dir"] is True
    assert ap["run_dir_template"] == "06_Training_Runs/{run_id}"
    assert ap["checkpoint_dir_template"] == "07_Checkpoints/{run_id}"
    assert ap["recovery_checkpoint_pattern"] == "recovery_epoch_{epoch:02d}.pt"
    assert ap["exact_leaf_names"] == {
        "best_checkpoint":"best_validation.pt",
        "checkpoint_hash_manifest":"CHECKPOINT_SHA256SUMS.txt",
        "completion_record":"completion_status.json",
        "environment":"environment.json",
        "epoch_metrics":"epoch_metrics.csv",
        "frozen_run_config_copy":"run_config.json",
        "run_manifest":"run_manifest.json",
        "training_log":"training_log.txt",
    }

    d = c["determinism"]
    assert d["cublas_workspace_config"] == ":4096:8"
    assert d["cudnn_deterministic"] is True
    assert d["cudnn_benchmark"] is False
    assert d["torch_deterministic_algorithms"] is True
    assert d["train_shuffle"] is True and d["validation_shuffle"] is False
    assert d["drop_last"] is False and d["num_workers"] == 2
    assert d["train_dataloader_generator_seed"] == "run_seed"
    assert d["worker_seed_rule"] == "torch.initial_seed() modulo 2^32"

    aug = c["augmentation"]
    assert aug["train"]["name"] == "Albumentations D4"
    assert aug["train"]["p"] == 1.0
    assert aug["train"]["synchronized_across_corresponding_tensors"] is True
    assert aug["validation"] == "none" and aug["test"] == "none"
    assert aug["photometric_or_sar_value_augmentation"] == "none"

    a = c["architecture"]
    assert a["family"] == "U-Net"
    if c["role"] == "teacher":
        assert c["condition_id"] == "TEACHER" and c["run_id"] == "SS-TEACHER-s00"
        assert c["seed"] == 0
        assert a == {
            "family":"U-Net","base":64,
            "input_channels":["VV_pre","VH_pre","VV_post","VH_post"],
            "output":"3_logits_N_P_F"
        }
        cp = c["checkpoint"]
        assert cp["metric"] == "source_validation_three_class_macro_iou"
        assert cp["prediction_rule"] == "softmax_argmax"
        assert cp["require_nonzero_union_for_all_classes"] is True
        assert cp["strict_improvement"] is True and cp["tie"] == "retain_earlier_epoch"
    else:
        cond = c["condition_id"]
        assert cond in EXPECTED_CONDITIONS and c["seed"] in EXPECTED_SEEDS
        assert c["run_id"] == f"SS-{cond}-s{c['seed']:02d}"
        assert a["base"] == 16 and a["input_channels"] == ["VV_post","VH_post"]
        if cond == "A5":
            assert a["output"] == "3_logits_N_P_F"
            assert c["checkpoint"]["prediction_rule"] == "softmax_p_F > 0.5"
        else:
            assert a["output"] == "1_flood_logit"
            assert c["checkpoint"]["prediction_rule"] == "sigmoid_probability > 0.5"
        assert c["checkpoint"]["metric"] == "source_validation_flood_micro_iou"
        assert c["checkpoint"]["threshold"] == 0.5
        assert c["checkpoint"]["strict_improvement"] is True
        assert c["checkpoint"]["tie"] == "retain_earlier_epoch"

        enabled, target, weight = _KD_EXPECTED[cond]
        kd = c["kd"]
        assert kd["enabled"] is enabled
        if enabled:
            assert kd["beta"] == 0.3 and kd["teacher_detached"] is True
            assert kd["target"] == target and kd["weight"] == weight
            assert c["teacher_dependency"]["required"] is True
            assert c["teacher_dependency"]["teacher_run_id"] == "SS-TEACHER-s00"
        else:
            assert kd["mode"] == "none" and kd["beta"] is None
            assert c["teacher_dependency"]["required"] is False
    return True

def validate_eval_config(c):
    assert c["schema_version"] == "semantic_shift_eval_config_v1"
    assert c["status"] == "FROZEN"
    assert c["implementation_binding"]["required_interface_version"] == INTERFACE_VERSION
    assert c["fixed_threshold"] == 0.5
    assert c["test_time_augmentation"] == "none"
    assert c["result_change_forbidden"] is True
    assert c["required_student_count"] == 40
    assert len(c["student_runs"]) == 40 and len(set(c["student_runs"])) == 40
    expected_runs = [f"SS-{cond}-s{seed:02d}" for cond in EXPECTED_CONDITIONS for seed in EXPECTED_SEEDS]
    assert c["student_runs"] == expected_runs
    _assert_common_preprocessing(c["preprocessing"])
    assert c["stage"] in {"E1","E2","E3"}
    expected_dir = {
        "E1":"08_Evaluation/E1_GEOID_SOURCE",
        "E2":"08_Evaluation/E2_GEOID_HELDOUT_2026",
        "E3":"08_Evaluation/E3_KURO_TEST_GRD",
    }[c["stage"]]
    assert c["output_stage_dir"] == expected_dir
    leaves = c["artifact_policy"]["exact_leaf_names"]
    assert leaves == {
        "artifact_hashes":"ARTIFACT_SHA256SUMS.txt",
        "domain_summary":"domain_summary.json",
        "evaluation_log":"evaluation_log.txt",
        "event_sufficient_statistics":"event_sufficient_statistics.csv",
    }
    if c["stage"] == "E1":
        assert c["dataset"]["expected_chips"] == 6192
        assert c["dataset"]["event_key_column"] == "event_aoi_id"
        assert c["teacher_descriptive"]["enabled"] is True
    elif c["stage"] == "E2":
        assert c["dataset"]["expected_chips"] == 16602
        assert c["dataset"]["event_key_column"] == "event_aoi_id"
        assert c["teacher_descriptive"]["enabled"] is True
    else:
        d = c["dataset"]
        assert d["dataset_revision"] is None
        assert d["expected_events"] == 43 and d["expected_samples"] == 67490
        assert d["channel_order"] == ["VV","VH"]
        assert d["native_chip_size"] == [224,224]
        assert d["ignored_label_values"] == [3]
        assert c["teacher_descriptive"]["enabled"] is False
    return True

def validate_analysis_config(c):
    assert c["schema_version"] == "semantic_shift_analysis_config_v1"
    assert c["status"] == "FROZEN"
    assert c["primary_endpoint"] == "event_macro_flood_iou_threshold_0.5"
    assert c["primary_contrasts"] == ["A3-A0","A3-B0"]
    assert c["primary_estimand_count"] == 6
    b = c["bootstrap"]
    assert b["type"] == "paired_crossed_seed_x_event"
    assert b["replicates"] == 50000 and b["seed"] == 20260910
    assert b["rng"] == "numpy.default_rng_PCG64"
    assert b["pointwise_percentile_ci"] == [0.025,0.975]
    assert b["familywise_two_sided_quantiles"] == [0.0041666667,0.9958333333]
    assert c["seed_set"] == [0,1,2,3,4]
    return True

def load_config(path):
    c = json.loads(Path(path).read_text())
    if c.get("schema_version") == "semantic_shift_run_config_v1":
        validate_run_config(c)
    elif c.get("schema_version") == "semantic_shift_eval_config_v1":
        validate_eval_config(c)
    elif c.get("schema_version") == "semantic_shift_analysis_config_v1":
        validate_analysis_config(c)
    else:
        raise AssertionError("Unknown Semantic Shift config schema")
    return c

<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Test index

**Generated from the test files.** Regenerate with `python tools/generate_test_index.py`.

Ordered as pytest collects them: files alphabetically, tests in declaration
order within each file.

## Tests do not produce figures

Worth stating plainly, because it is a natural assumption. The suite renders
figures in memory to check that they draw, and closes them; it writes **no
files at all**. Every PNG and CSV comes from the example scripts:

```bash
python examples/run_all.py        # 54 figures, 45 tables, 3 run reports
```

| Example | Produces |
|---|---|
| `example_01_pipeline.py` | signal, phase-amplitude and quality figures; generator validation and corpus tables |
| `example_02_statespace.py` | attractor figures for every constructor, route comparison, embedding curves |
| `example_03_recurrence.py` | recurrence, cross- and joint-recurrence plots, density profiles, threshold diagnostics |
| `example_04_windowed.py` | window coverage and series, per-window recurrence plots |
| `example_05_comparability.py` | comparability reports, group comparison, feature matrix |
| `example_06_cookbook.py` | 39 figures into `cookbook_out/`, one per recipe |

---

## Summary

| # | File | Covers | Decisions | Tests |
|---|---|---|---|---|
| 1 | `test_capabilities.py` | Capability system | DD-03 | 3 |
| 2 | `test_classify.py` | test_classify.py |  | 25 |
| 3 | `test_cli.py` | test_cli.py |  | 3 |
| 4 | `test_comparability.py` | Comparability and groups | DD-42, DD-43, DD-44 | 29 |
| 5 | `test_docs.py` | The living documents | DD-45 | 18 |
| 6 | `test_dynamics.py` | test_dynamics.py |  | 63 |
| 7 | `test_export.py` | CSV export | DD-10 | 4 |
| 8 | `test_images.py` | test_images.py |  | 16 |
| 9 | `test_ingest.py` | Ingestion | DD-08, DD-06 | 12 |
| 10 | `test_meta.py` | test_meta.py |  | 6 |
| 11 | `test_parallel.py` | test_parallel.py |  | 7 |
| 12 | `test_preprocess.py` | Filtering and Hilbert | DD-11, DD-12 | 10 |
| 13 | `test_realdata.py` | test_realdata.py |  | 15 |
| 14 | `test_recording.py` | The Recording object | DD-07, DD-04 | 5 |
| 15 | `test_recurrence.py` | Recurrence structures | DD-19, DD-20, DD-33, DD-34, DD-36 | 41 |
| 16 | `test_report.py` | Run reports | DD-37 | 8 |
| 17 | `test_rqa.py` | test_rqa.py |  | 26 |
| 18 | `test_scaling.py` | The scaling policy | DD-02 | 8 |
| 19 | `test_signals.py` | The synthetic generator | DD-13, DD-48 | 29 |
| 20 | `test_statespace.py` | State space construction | DD-21, DD-23, DD-25, DD-28 | 50 |
| 21 | `test_threshold.py` | Threshold selection | DD-32, DD-35 | 18 |
| 22 | `test_validation.py` | test_validation.py |  | 26 |
| 23 | `test_viz.py` | Figures | DD-27, DD-30, DD-31, DD-38, DD-49..52 | 39 |
| 24 | `test_windowed.py` | Windowed analysis | DD-39, DD-40, DD-41 | 49 |
| | | | **total** | **510 functions** |

Parametrised tests expand to more cases; run `pytest -q` for the count.

---

## 1. `test_capabilities.py` — Capability system

That a stage refuses to run without the data it needs, and says which capability is missing.

Decisions: DD-03

| Test | What it checks |
|---|---|
| `test_flags_compose` | Flags compose. |
| `test_require_raises_with_named_missing` | Require raises with named missing. |
| `test_stage_refuses_without_capability` | Stage refuses without capability. |

## 2. `test_classify.py` — test_classify.py

| Test | What it checks |
|---|---|
| `test_families_select_the_right_columns` | Families select the right columns. |
| `test_bookkeeping_columns_are_never_features` | Bookkeeping columns are never features. |
| `test_unknown_family_is_rejected` | Unknown family is rejected. |
| `test_subject_level_reduces_to_one_row_each` | Subject level reduces to one row each. |
| `test_window_level_keeps_the_rows_and_names_their_subject` | Window level keeps the rows and names their subject. |
| `test_windows_of_a_subject_never_straddle_a_fold` | The result this whole module depends on. If a subject's windows appear in training and testing, the model recognises the subject, not the group. |
| `test_shuffled_labels_score_at_chance` | A pipeline that leaks would score above chance on random labels. This is the check that a passing AUC is worth anything at all [DD-83, DD-84]. |
| `test_every_requested_score_is_present` | Every requested score is present. |
| `test_imbalance_is_reported_with_a_baseline` | DD-85: accuracy alone on 10 against 6 is not interpretable. |
| `test_repeats_give_a_spread` | DD-86: one split is an anecdote. |
| `test_small_sample_is_flagged` | Small sample is flagged. |
| `test_more_features_than_subjects_is_flagged` | More features than subjects is flagged. |
| `test_report_exports_and_summarises` | Report exports and summarises. |
| `test_permutation_null_is_centred_at_chance` | Permutation null is centred at chance. |
| `test_permutation_shuffles_whole_subjects` | DD-87: shuffling rows rather than subjects would leave a subject with two labels and make the null far too easy to beat. |
| `test_needs_exactly_two_groups` | Needs exactly two groups. |
| `test_missing_label_is_reported` | Missing label is reported. |
| `test_models_run` _(×2)_ | Models run. |
| `test_unknown_model_is_rejected` | Unknown model is rejected. |
| `test_whole_recording_is_a_single_window` | The whole-signal case is the same code with one window per subject. |
| `test_classify_timecourse_returns_one_honest_row_per_window` | Classify timecourse returns one honest row per window. |
| `test_classical_family_finds_coupling_indices_output` | select_features('classical') must see what coupling_indices writes. |
| `test_degenerate_test_folds_are_flagged_not_scored` | A single-class test fold gets NaN scores and a flag, and no sklearn warning; its predictions still count in the repeat-level scores. |
| `test_permutation_p_is_independent_of_n_jobs` | The shuffles are seeded by position, so p must not depend on workers. |
| `test_classify_timecourse_is_independent_of_n_jobs` | Classify timecourse is independent of n jobs. |

## 3. `test_cli.py` — test_cli.py

| Test | What it checks |
|---|---|
| `test_doctor_lists_the_core_and_the_optional_backends` | Doctor lists the core and the optional backends. |
| `test_figures_lists_the_catalogue` | Figures lists the catalogue. |
| `test_generate_writes_the_signals_and_their_ground_truth` | Before 0.25 it wrote only the ground-truth table and discarded the signals it had generated. |

## 4. `test_comparability.py` — Comparability and groups

That the three silent failure modes are detected, and that groups separate window by window.

Decisions: DD-42, DD-43, DD-44

| Test | What it checks |
|---|---|
| `test_n_windows_on_unequal_records_is_flagged` | Failure mode 2: 'window 0' stops meaning the same thing. |
| `test_n_windows_warns_in_batch` | N windows warns in batch. |
| `test_fixed_window_size_is_comparable` | Fixed window size is comparable. |
| `test_different_dimensions_are_caught` | Failure mode 1: auto embedding puts subjects in different spaces. |
| `test_different_effective_fs_is_caught` | Failure mode 3: asking for the same point count from records of different length decimates them by different factors, so their effective sampling rates diverge. Since DD-47 the rate is carried on t... |
| `test_contract_declares_only_what_is_set` | Contract declares only what is set. |
| `test_contract_violation_is_named` | Contract violation is named. |
| `test_strict_contract_raises` | Strict contract raises. |
| `test_contract_from_a_reference` | Contract from a reference. |
| `test_tolerance_allows_small_numeric_drift` | Tolerance allows small numeric drift. |
| `test_describe_units_tabulates` | Describe units tabulates. |
| `test_matched_rate_licenses_ratios_not_the_rate` | Matched rate licenses ratios not the rate. |
| `test_unmatched_length_forbids_length_metrics` | Unmatched length forbids length metrics. |
| `test_metric_requirements_cover_every_family` | Metric requirements cover every family. |
| `test_group_comparison_is_window_by_window` | DD-44: window k of one subject against window k of another. |
| `test_group_comparison_separates_the_groups` | Group comparison separates the groups. |
| `test_group_comparison_reports_multiplicity` | Group comparison reports multiplicity. |
| `test_group_comparison_needs_exactly_two_groups` | Group comparison needs exactly two groups. |
| `test_pooled_comparison_with_by_none` | Pooled comparison with by none. |
| `test_feature_matrix_shape` | Feature matrix shape. |
| `test_feature_matrix_rejects_unknown_column` | Feature matrix rejects unknown column. |
| `test_rank_metrics_orders_by_separation` | Rank metrics orders by separation. |
| `test_comparability_figures_render` | Comparability figures render. |
| `test_group_figure_needs_one_metric` | Group figure needs one metric. |
| `test_comparability_figure_carries_a_legend` | Four statuses in colour alone is not readable [DD-55]. |
| `test_absent_axes_are_annotated` | With per-window thresholds epsilon is absent, not violated, and the figure must say which [DD-55]. |
| `test_group_timecourse_finds_a_late_effect_and_holds_its_level` | p_peak catches an effect confined to late windows; a null metric stays null. |
| `test_group_timecourse_needs_two_groups_and_known_columns` | Group timecourse needs two groups and known columns. |
| `test_group_timecourse_p_is_independent_of_n_jobs` | Permutations are drawn before any split, so p_peak/p_trend must match. |

## 5. `test_docs.py` — The living documents

That the documents have not drifted from the code: decision index, citations, module map, figure catalogue, counts, API coverage.

Decisions: DD-45

| Test | What it checks |
|---|---|
| `test_decision_index_matches_sections` | Decision index matches sections. |
| `test_decisions_are_numbered_without_gaps` | Decisions are numbered without gaps. |
| `test_every_decision_cited_in_source_exists` | A dangling [DD-nn] in a docstring is a promise the documents do not keep. |
| `test_major_decisions_are_cited_somewhere_in_the_source` | A decision about the code should be traceable from the code. |
| `test_every_module_appears_in_the_architecture_map` | Every module appears in the architecture map. |
| `test_architecture_quotes_the_real_test_count` | A stale count is the first sign a document stopped being maintained. |
| `test_figure_catalogue_lists_every_registered_figure` | Figure catalogue lists every registered figure. |
| `test_example_scripts_referenced_in_the_readme_exist` | Example scripts referenced in the readme exist. |
| `test_every_example_script_is_referenced` | Every example script is referenced. |
| `test_examples_document_covers_the_public_api` | Every public entry point should appear somewhere in the examples. |
| `test_changelog_mentions_the_current_version` | Changelog mentions the current version. |
| `test_cookbook_exercises_the_whole_public_api` | The cookbook is the acceptance test for the API surface [DD-45]. |
| `test_cookbook_is_split_into_spyder_cells` | Cookbook is split into spyder cells. |
| `test_test_index_is_current` | docs/TESTS.md is generated; a stale one is worse than none [DD-45]. |
| `test_example_index_lists_every_script` | docs/EXAMPLE_INDEX.md must cover each example [DD-45]. |
| `test_example_index_names_the_figures_the_scripts_save` | Every filename an example writes must be described in the index. Catches the drift that matters: a figure added to a script and never documented, or one documented after being removed. |
| `test_documented_metrics_all_exist` | A metric named in the documents must exist in the code [DD-45]. This check exists because it did not. Two implementations of the RQA layer both passed their own suites while exporting different met... |
| `test_the_version_is_not_stale` | It sat at 0.1.0.dev0 through nineteen delivered builds. An installed copy could not be told apart from any other, which is exactly what a version string exists to prevent. |

## 6. `test_dynamics.py` — test_dynamics.py

| Test | What it checks |
|---|---|
| `test_scaling_region_recovers_a_known_slope` | Scaling region recovers a known slope. |
| `test_scaling_region_avoids_a_bent_tail` | A curve that is straight then saturates: the fit must take the straight part, which is the whole point of locating a region rather than fitting everything [DD-61]. |
| `test_narrow_region_is_flagged` | Narrow region is flagged. |
| `test_fixed_region_honours_the_window` | Fixed region honours the window. |
| `test_local_slopes_are_constant_for_a_line` | Local slopes are constant for a line. |
| `test_lorenz_correlation_dimension` | Lorenz correlation dimension. |
| `test_rossler_correlation_dimension` | Rossler correlation dimension. |
| `test_dimension_orders_the_systems` | Rossler is nearly a two-dimensional band; Lorenz is thicker. |
| `test_correlation_sum_is_monotone` | Correlation sum is monotone. |
| `test_dimension_accepts_a_user_region` | Dimension accepts a user region. |
| `test_lorenz_lyapunov` | Reference 0.906 per second. The curve has to be long enough to contain a linear region [DD-94], and that matters more here than the sampling does [DD-101]: measured on this space, max_steps=150 giv... |
| `test_lyapunov_units_convert_by_the_sampling_rate` | DD-62: a rate without its unit is not a number anyone can use. |
| `test_an_unflagged_lyapunov_exponent_can_be_trusted` | DD-94 and DD-101 together, as a contract rather than an anecdote. The guards exist so that a value without a warning is a value the data determines. Before averaging, one sampling at max_steps=1200... |
| `test_divergence_curve_rises` | Divergence curve rises. |
| `test_noiseless_orbit_is_flagged_as_numerically_degenerate` | On a sampled circle the nearest neighbours sit at machine precision -- 1.3e-15 measured -- so the divergence curve tracks floating-point noise through a logarithm. The estimator cannot tell that ap... |
| `test_a_real_attractor_can_be_estimated_without_a_flag` | Chaos with a curve long enough to hold a linear region and short enough not to saturate. If nothing here is clean, the estimator is unusable. |
| `test_unknown_units_rejected` | Unknown units rejected. |
| `test_k2_is_positive_for_chaos` | K2 is positive for chaos. |
| `test_k2_orders_chaos_above_a_noisy_limit_cycle` | Compared in per-sample units, so the two sampling rates do not enter. |
| `test_k2_refuses_a_plot_too_sparse_to_fit` | K2 comes from the decay rate of the diagonal length distribution. On a plot with almost no lines there is no decay to fit, and refusing is better than returning a slope through three points. |
| `test_k2_reports_how_much_evidence_it_had` | K2 reports how much evidence it had. |
| `test_dfa_exponent_of_coloured_noise` _(×3)_ | Dfa exponent of coloured noise. |
| `test_hurst_matches_dfa` | Hurst matches dfa. |
| `test_higuchi_dimension_bounds` | Higuchi dimension bounds. |
| `test_permutation_entropy_is_maximal_for_noise` | Permutation entropy is maximal for noise. |
| `test_sample_entropy_orders_noise_above_a_sine` | Sample entropy orders noise above a sine. |
| `test_spectral_entropy_is_low_for_a_pure_tone` | Spectral entropy is low for a pure tone. |
| `test_short_series_is_refused` | Short series is refused. |
| `test_invariant_frame_and_summary` | Invariant frame and summary. |
| `test_hjorth_mobility_matches_theory` | For a sinusoid at f, mobility is 2*pi*f/fs exactly. A rare closed form among these measures, and worth pinning. |
| `test_hjorth_mobility_reports_its_frequency_equivalent` | Hjorth mobility reports its frequency equivalent. |
| `test_lempel_ziv_is_maximal_for_noise` | Lempel ziv is maximal for noise. |
| `test_svd_entropy_is_maximal_for_noise` | Svd entropy is maximal for noise. |
| `test_katz_and_petrosian_order_smooth_below_rough` | Katz and petrosian order smooth below rough. |
| `test_higuchi_is_anchored_and_stays_in_range` | DD-73: floating the fit put a smoothed envelope at 3.16, outside the [1, 2] the dimension is defined on. |
| `test_higuchi_warns_when_there_is_no_scaling_range` | Higuchi warns when there is no scaling range. |
| `test_phase_blocks_contribute_their_instantaneous_frequency` | DD-71: the wrapped angle jumps at +-pi and DFA read 0.13; the unwrapped angle is a ramp and DFA read 2.01. The increment is the stationary quantity. |
| `test_dynamics_measures_names_the_block` | Dynamics measures names the block. |
| `test_unknown_measure_is_rejected` | Unknown measure is rejected. |
| `test_invariants_are_opt_in` | DD-72: cheap by default, expensive on request. |
| `test_dynamics_enter_the_windowed_table` | Dynamics enter the windowed table. |
| `test_block_invariants_estimate_each_signal_on_its_own` | DD-93. The earlier draft computed a univariate Lyapunov exponent and correlation dimension per component as well as a multivariate one, and both are useful: one describes the joint trajectory, the ... |
| `test_block_and_trajectory_invariants_are_different_quantities` | Block and trajectory invariants are different quantities. |
| `test_unknown_block_invariant_is_rejected` | Unknown block invariant is rejected. |
| `test_a_block_too_short_to_embed_warns_rather_than_returning_nan_silently` | A block too short to embed warns rather than returning nan silently. |
| `test_auto_theiler_scales_with_the_embedding` | DD-96. Points closer in time than the embedding window share coordinates, so counting them as neighbours makes the cloud look one-dimensional exactly where D2 is read. |
| `test_correlation_dimension_of_lorenz_from_a_delay_embedding` | The reference case, as a contract over four trajectories [DD-96, DD-121]. Initial conditions 1e-9 apart stand in for four machines: chaos turns a difference in rounding -- the fused multiply-add of... |
| `test_the_lyapunov_exponent_from_a_series_keeps_its_contract` | DD-101, from a delay embedding of one coordinate. Which reference points are drawn was deciding the answer: twelve streams over identical data gave 0.501 to 1.366, eleven of them unflagged. Three s... |
| `test_the_sampling_spread_is_reported` | An estimate that moves with the draw must say so [DD-101]. |
| `test_one_sampling_is_the_old_behaviour` | One sampling is the old behaviour. |
| `test_the_embedding_used_is_reported` | A result nobody can audit is not a result. |
| `test_series_invariants_accept_explicit_parameters` | Series invariants accept explicit parameters. |
| `test_series_invariants_reject_an_unknown_measure` | Series invariants reject an unknown measure. |
| `test_a_short_series_is_refused` | A short series is refused. |
| `test_a_smooth_series_reports_dimension_one_and_says_why` | DD-99. Higuchi, Katz, Petrosian and the ordinal entropies read point-to-point structure. Given thirty samples per cycle of the fastest content they report, correctly, a smooth line -- and across 98... |
| `test_white_noise_needs_no_decimation` | The reference scale: one sample per event, which is where these measures are defined. |
| `test_auto_fine_scale_revives_the_flat_columns` | Auto fine scale revives the flat columns. |
| `test_the_factor_and_its_reason_are_recorded` | A decimation nobody can see is a decimation nobody can check [DD-99]. |
| `test_dfa_and_hjorth_see_the_whole_series` | Only the fine-grained measures are decimated; a scaling exponent needs every scale it can get. |
| `test_a_correlation_sum_that_never_bends_is_flagged` | DD-100. A phase circle crossed with an amplitude is a 2-torus by construction, so D2 comes back at 1.98 with R2 = 1.0000 on every recording. That is the dimension of the construction, not of the data. |
| `test_a_real_attractor_has_a_region_to_find` | The guard must not fire on a dimension that was measured [DD-100]. |
| `test_phase_block_series_has_no_spurious_leading_sample` | The phase increment series starts with a real increment, not a zero. |
| `test_fine_scale_decimation_leaves_enough_samples` | DD-99, revised in 0.25. A slow series in a short window asked for a decimation that left nine samples, and every fractal and entropy measure failed. The factor is now capped, the cap is announced, ... |

## 7. `test_export.py` — CSV export

That every table gets its provenance and environment sidecars.

Decisions: DD-10

| Test | What it checks |
|---|---|
| `test_export_frame_writes_sidecars` | Export frame writes sidecars. |
| `test_environment_stamp_has_version` | Environment stamp has version. |
| `test_export_recording_tables` | Export recording tables. |
| `test_csv_exporter_accumulates` | Csv exporter accumulates. |

## 8. `test_images.py` — test_images.py

| Test | What it checks |
|---|---|
| `test_image_is_a_density_map_in_the_unit_interval` | DD-90: every output pixel is the mean of the cells it covers, so the value is a local recurrence density, not a thresholded pixel. |
| `test_pooling_preserves_density_where_nearest_neighbour_would_not` | The argument for pooling, made as a measurement: a plot reduced by a large factor keeps its rate under mean pooling, and would keep only one cell in factor-squared under resampling. |
| `test_every_image_has_the_size_that_was_asked_for` | Recordings of different length must still give one shape, or a network cannot take them [DD-90]. |
| `test_unknown_pooling_is_rejected` | Unknown pooling is rejected. |
| `test_the_brightness_scale_is_shared_across_the_batch` | DD-91: per-image normalisation rescales away exactly the differences the images exist to show, because every recording is pinned to the same rate. |
| `test_per_image_normalisation_is_available_and_marked` | Per image normalisation is available and marked. |
| `test_a_fixed_scale_can_be_shared_between_batches` | A fixed scale can be shared between batches. |
| `test_metadata_describes_how_each_image_was_made` | Metadata describes how each image was made. |
| `test_npy_output_needs_no_image_library` | PNG writing depends on Pillow, which is not always healthy. The raw density can always be written. |
| `test_unknown_format_is_rejected` | Unknown format is rejected. |
| `test_decimating_records_what_it_did` | DD-92: the two routes answer different questions and images made one way are not comparable with the other, so which was used is recorded. |
| `test_an_attractor_image_counts_rather_than_marks` | DD-103. Drawing 34 000 points on a 256-pixel grid saturates into a silhouette: the shape survives and how often the trajectory visits each part does not. The same failure as thresholding a recurren... |
| `test_the_projection_is_recorded` | A picture of a three-dimensional space is two of its coordinates, and which two is not a detail [DD-103]. |
| `test_an_impossible_projection_is_refused` | An impossible projection is refused. |
| `test_attractor_export_shares_its_scale` | DD-91 again: per-image normalisation would rescale away the differences between subjects. |
| `test_an_outlying_burst_does_not_squeeze_the_trajectory` | DD-50: robust limits, or one burst puts everything in a few pixels. |

## 9. `test_ingest.py` — Ingestion

The input shapes, role assignment, quality checks, and the errors that name their own fix.

Decisions: DD-08, DD-06

| Test | What it checks |
|---|---|
| `test_1d_array` | 1d array. |
| `test_2d_array_with_labels` | 2d array with labels. |
| `test_transposed_input_is_rejected_with_advice` | Transposed input is rejected with advice. |
| `test_dict_of_hilbert_components_sets_roles` | Dict of hilbert components sets roles. |
| `test_explicit_roles_beat_name_guessing` | Explicit roles beat name guessing. |
| `test_complex_channel_is_split` | Complex channel is split. |
| `test_tuple_phase_amplitude` | Tuple phase amplitude. |
| `test_statespace_kind` | Statespace kind. |
| `test_missing_fs_is_an_error` | Missing fs is an error. |
| `test_recording_passthrough_is_idempotent` | Recording passthrough is idempotent. |
| `test_csv_roundtrip` | Csv roundtrip. |
| `test_quality_flags_constant_channel` | Quality flags constant channel. |

## 10. `test_meta.py` — test_meta.py

| Test | What it checks |
|---|---|
| `test_the_meta_space_keeps_measurements_and_drops_bookkeeping` | The meta space keeps measurements and drops bookkeeping. |
| `test_the_automatic_theiler_window_excludes_overlapping_windows` _(×3)_ | Windows fewer than size/step positions apart share samples. |
| `test_a_regime_alternation_recurs_at_its_period` | Ground truth: two regimes alternating with a period of 6 windows give a meta-RP that is full on diagonals at multiples of 6 and empty halfway. |
| `test_meta_recurrence_from_a_windowed_analysis` | Meta recurrence from a windowed analysis. |
| `test_missing_windows_are_dropped_with_a_warning` | Missing windows are dropped with a warning. |
| `test_cross_and_joint_meta_recurrence` | Cross and joint meta recurrence. |

## 11. `test_parallel.py` — test_parallel.py

| Test | What it checks |
|---|---|
| `test_job_count_resolution` | Job count resolution. |
| `test_unknown_backend_is_rejected` | Unknown backend is rejected. |
| `test_parallel_map_preserves_input_order` | Parallel map preserves input order. |
| `test_random_streams_depend_on_position_not_scheduling` | DD-81. Each item's stream is derived from its position before any work starts, so two runs at different job counts must draw the same numbers. |
| `test_batch_results_do_not_depend_on_n_jobs` | The promise of DD-05, checked on the real pipeline rather than argued. |
| `test_batch_worker_is_picklable` | DD-81: the process backend pickles the callable, so it cannot be a lambda. This is the mistake the first implementation made. |
| `test_serial_backend_is_always_available` | joblib is an optional dependency; asking for one worker must never reach it. |

## 12. `test_preprocess.py` — Filtering and Hilbert

Filter order from cycle counts, band isolation, edge policies, and the full chain end to end.

Decisions: DD-11, DD-12

| Test | What it checks |
|---|---|
| `test_auto_order_guarantees_minimum_cycles` | Auto order guarantees minimum cycles. |
| `test_band_outside_nyquist_rejected` | Band outside nyquist rejected. |
| `test_bandpass_isolates_the_band` | Bandpass isolates the band. |
| `test_bandpass_records_edge_cost` | Bandpass records edge cost. |
| `test_analytic_produces_all_three_components` | Analytic produces all three components. |
| `test_analytic_phase_is_wrapped_and_amplitude_positive` | Analytic phase is wrapped and amplitude positive. |
| `test_edge_trim_shortens_and_is_recorded` | Edge trim shortens and is recorded. |
| `test_mirror_policy_preserves_length` | Mirror policy preserves length. |
| `test_resample_changes_fs_and_length` | Resample changes fs and length. |
| `test_full_chain_recovers_known_pac` | End to end: generate -> ingest -> filter -> Hilbert -> canonical index. |

## 13. `test_realdata.py` — test_realdata.py

| Test | What it checks |
|---|---|
| `test_edf_round_trip` | Edf round trip. |
| `test_units_come_from_the_file` | DD-88: the physical unit is a fact about the recording. |
| `test_metadata_and_annotations_are_kept` | Metadata and annotations are kept. |
| `test_mixed_rates_are_refused_by_name` | DD-88: a real montage mixes rates, and the message must say which. |
| `test_channel_selection_resolves_mixed_rates` | Channel selection resolves mixed rates. |
| `test_unknown_channel_is_reported_with_what_is_there` | Unknown channel is reported with what is there. |
| `test_an_edf_reaches_a_state_space` | The point of a reader: that the rest of the library accepts what it gives. |
| `test_corpus_finds_subjects_and_joins_labels` | Corpus finds subjects and joins labels. |
| `test_corpus_reports_the_join_rather_than_hiding_it` | DD-89: a row with no file is the commonest way a study goes wrong. |
| `test_corpus_frame_lists_everything` | Corpus frame lists everything. |
| `test_corpus_holds_paths_not_data` | A corpus of eight-minute recordings cannot be loaded at once. |
| `test_empty_directory_is_reported` | Empty directory is reported. |
| `test_two_files_for_one_subject_is_refused` | Two files for one subject is refused. |
| `test_ambiguous_label_column_asks` | Ambiguous label column asks. |
| `test_corpus_labels_feed_the_classifier` | The chain the whole module exists for: files on disk to a group label. |

## 14. `test_recording.py` — The Recording object

Copy-on-write sharing, read-only buffers, provenance accumulation.

Decisions: DD-07, DD-04

| Test | What it checks |
|---|---|
| `test_copy_on_write_shares_buffers` | Copy on write shares buffers. |
| `test_arrays_are_read_only` | Arrays are read only. |
| `test_length_mismatch_rejected` | Length mismatch rejected. |
| `test_provenance_accumulates` | Provenance accumulates. |
| `test_channel_frame_is_exportable` | Channel frame is exportable. |

## 15. `test_recurrence.py` — Recurrence structures

Distance kernels, streaming equivalence bit for bit, Theiler bands, RP/CRP/JRP, memory refusal.

Decisions: DD-19, DD-20, DD-33, DD-34, DD-36

| Test | What it checks |
|---|---|
| `test_expanded_norm_matches_naive_broadcasting` | DD-33: the BLAS expansion must be numerically identical. |
| `test_squared_distances_are_non_negative` | Squared distances are non negative. |
| `test_all_metrics_build_a_plot` _(×4)_ | All metrics build a plot. |
| `test_tiled_equals_dense_bit_for_bit` | The whole long-signal strategy depends on this being exact. |
| `test_recurrence_rate_is_tile_size_independent` _(×4)_ | Recurrence rate is tile size independent. |
| `test_sparse_matches_dense` | Sparse matches dense. |
| `test_plot_is_symmetric` | Plot is symmetric. |
| `test_theiler_zero_keeps_the_line_of_identity` | Corrected convention [DD-19]: theiler=0 excludes nothing. |
| `test_theiler_one_removes_the_line_of_identity` | Theiler one removes the line of identity. |
| `test_theiler_one_removes_nothing_but_the_line_of_identity` | Standard convention [DD-115]: the first off-diagonals survive at theiler=1, as in the CRP Toolbox, pyunicorn and PyRQA. Before 0.25 they were removed too, so a window of 1 here meant a window of 2 ... |
| `test_theiler_band_is_exactly_excluded` _(×4)_ | Exactly the band \|i - j\| < theiler is removed [DD-115]: not one cell more, not one less. An epsilon larger than the attractor makes every pair recurrent, so the matrix must be 0 inside the band a... |
| `test_recurrence_rate_matches_the_target` | Recurrence rate matches the target. |
| `test_rate_denominator_excludes_theiler` | The reported rate must be comparable with the threshold's estimate. |
| `test_fixed_threshold_shortcut` | Fixed threshold shortcut. |
| `test_fan_makes_an_asymmetric_plot` | Fan makes an asymmetric plot. |
| `test_per_block_threshold_uses_chebyshev_across_blocks` | Per block threshold uses chebyshev across blocks. |
| `test_sine_gives_diagonals_at_multiples_of_its_period` | Ground truth. A circle traced at frequency f0 recurs with itself every period, so the recurrence plot must be dense on the diagonals at integer multiples of fs/f0 and empty between them. |
| `test_crp_is_rectangular_and_asymmetric` | Crp is rectangular and asymmetric. |
| `test_crp_rejects_mismatched_dimensions` | Crp rejects mismatched dimensions. |
| `test_jrp_allows_different_dimensions` | DD-20: this is the whole point of the joint plot. |
| `test_jrp_uses_independent_thresholds` | Jrp uses independent thresholds. |
| `test_jrp_is_the_and_of_its_parts` | Jrp is the and of its parts. |
| `test_jrp_rate_never_exceeds_its_parts` | Jrp rate never exceeds its parts. |
| `test_jrp_independence_ratio_is_one_for_independent_systems` | Jrp independence ratio is one for independent systems. |
| `test_jrp_rejects_misaligned_records` | Jrp rejects misaligned records. |
| `test_jrp_needs_two_subsystems` | Jrp needs two subsystems. |
| `test_plan_refuses_an_impossible_dense_matrix` | Plan refuses an impossible dense matrix. |
| `test_plan_falls_back_to_streaming` | Plan falls back to streaming. |
| `test_plan_uses_memory_when_it_fits` | Plan uses memory when it fits. |
| `test_materialise_refuses_over_budget` | Built happily while the budget allowed it; refuses once it does not. |
| `test_describe_and_summary` | Describe and summary. |
| `test_density_profile_streams` | Density profile streams. |
| `test_provenance_carries_through` | Provenance carries through. |
| `test_single_precision_is_translation_invariant` | Centring before the norm expansion: an offset must not change the plot. Before DD-110 an offset of 1000 in every coordinate moved the recurrence rate of a single-precision plot from 0.051 to 0.082. |
| `test_every_metric_matches_scipy_on_an_offset_cloud` _(×4)_ | An independent reference for each distance. Coordinates far from the origin matter: centring them is exact for the three translation-invariant metrics and wrong for cosine, which was centred anyway... |
| `test_a_plot_inherits_the_theiler_window_of_its_threshold` | A plot inherits the theiler window of its threshold. |
| `test_a_mismatched_theiler_window_or_metric_warns` | Measured before 0.25: estimated at 1, used at 50, the rate came out at 0.024 for a target of 0.05 with no warning. |
| `test_threshold_and_plot_defaults_agree` | Threshold and plot defaults agree. |
| `test_streaming_is_called_stream_and_the_old_name_still_works` | Streaming is called stream and the old name still works. |
| `test_epsilon_at_rounding_level_is_reported` | A sine with 2.5% exactly repeated pairs and a target of 0.02. |
| `test_single_precision_reaches_every_structure` | Single precision reaches every structure. |

## 16. `test_report.py` — Run reports

Sections, artefact registration, rendering, and that the report is in English.

Decisions: DD-37

| Test | What it checks |
|---|---|
| `test_sections_and_notes` | Sections and notes. |
| `test_export_registers_the_table` | Export registers the table. |
| `test_figure_registration` | Figure registration. |
| `test_warnings_are_collected` | Warnings are collected. |
| `test_write_produces_txt_and_index` | Write produces txt and index. |
| `test_report_records_parameters` | Report records parameters. |
| `test_artifact_frame_columns` | Artifact frame columns. |
| `test_report_text_is_english` | Report text is english. |

## 17. `test_rqa.py` — test_rqa.py

| Test | What it checks |
|---|---|
| `test_run_carry` _(×7)_ | The carry across a tile boundary, on cases checkable by hand [DD-56]. |
| `test_streaming_histogram_equals_dense` _(×3)_ | The whole long-signal strategy rests on this being exact [DD-56]. |
| `test_histogram_rate_matches_the_matrix` | Histogram rate matches the matrix. |
| `test_every_point_is_on_a_line_of_length_one_or_more` | Conservation: the diagonal histogram must account for every recurrent point, since every point lies on a diagonal run of at least one. |
| `test_white_noise_determinism_is_twice_the_rate` | For an independent recurrence matrix of density p, the fraction of points on diagonal lines of length >= 2 is 1 - (1 - p)^2, about 2p. **At l_min=2, which is where that identity is defined.** The l... |
| `test_periodic_signal_is_deterministic_and_spans_the_record` | Periodic signal is deterministic and spans the record. |
| `test_lorenz_is_strongly_deterministic` | Lorenz is strongly deterministic. |
| `test_determinism_orders_the_three_reference_systems` | Noise, a chaotic attractor and a periodic orbit must order correctly at a matched recurrence rate. |
| `test_denominator_convention_changes_det` | DD-58: the choice is real, systematic, and previously silent. |
| `test_unknown_denominator_is_rejected` | Unknown denominator is rejected. |
| `test_l_min_filters_short_lines` | L min filters short lines. |
| `test_normalised_entropy_is_bounded` | Normalised entropy is bounded. |
| `test_metric_selection` | Metric selection. |
| `test_all_declared_metrics_are_produced` | All declared metrics are produced. |
| `test_as_frame_carries_the_conventions` | As frame carries the conventions. |
| `test_diagonal_profile_finds_a_known_lag` | A trajectory against a shifted copy of itself must peak at the shift. |
| `test_crqa_reports_lag_and_asymmetry` | Crqa reports lag and asymmetry. |
| `test_rqa_enters_the_windowed_table` | Rqa enters the windowed table. |
| `test_matrix_convenience_methods` | Matrix convenience methods. |
| `test_rqa_never_needs_the_matrix` | Streaming all the way: a structure that is never materialised. |
| `test_saturated_determinism_is_flagged` | DD-98. DET pinned against its ceiling cannot separate anything, and on a smooth trajectory that is where it sits with the conventional floor of 2. Both floors now default to 8. Which failure mode a... |
| `test_the_defaults_departed_from_the_classical_two` | DD-98. Both floors default to 8 rather than the classical 2, because on a real corpus 2 saturates both metrics: DET at 0.9965 with a spread of 0.0005, LAM at 0.9983. The first version of this decis... |
| `test_a_crushed_metric_is_flagged_as_loudly_as_a_saturated_one` | A crushed metric is flagged as loudly as a saturated one. |
| `test_the_sweep_answers_the_floor_question_from_the_data` | DD-98: the histogram already holds every threshold, so choosing well costs milliseconds against the minutes the plot took. |
| `test_the_sweep_does_not_warn_about_the_floors_it_is_testing` | DD-98. The sweep exists to find the floor at which a metric has room, so its whole job is to evaluate floors that do not work. Warning about each one produced as many warnings as floors, per subjec... |
| `test_normalised_entropy_uses_admissible_lengths` | One rare long line must not move ENTR_norm by itself. |

## 18. `test_scaling.py` — The scaling policy

The analytic result the default rests on, and that the balance holds across five orders of magnitude of gain.

Decisions: DD-02

| Test | What it checks |
|---|---|
| `test_unit_circle_rms_is_sqrt_two` | The analytic result the default policy is built on. |
| `test_scalar_rms_is_sigma_sqrt_two` | Scalar rms is sigma sqrt two. |
| `test_sampled_and_theoretical_rms_agree` | Sampled and theoretical rms agree. |
| `test_rms_balanced_equalises_regardless_of_amplitude_scale` _(×5)_ | Rms balanced equalises regardless of amplitude scale. |
| `test_unscaled_blocks_are_dominated_and_flagged` | Unscaled blocks are dominated and flagged. |
| `test_lambda_controls_share_exactly` _(×5)_ | Lambda controls share exactly. |
| `test_lambda_rejects_bad_values` | Lambda rejects bad values. |
| `test_gain_invariance_of_the_default` | A global gain change must not alter the geometry. |

## 19. `test_signals.py` — The synthetic generator

That the generated coupling is monotone in alpha against the canonical indices, and reproducible.

Decisions: DD-13, DD-48

| Test | What it checks |
|---|---|
| `test_colored_noise_slopes` | Colored noise slopes. |
| `test_all_modalities_generate` _(×5)_ | All modalities generate. |
| `test_pac_strength_tracks_alpha` | H1.1 equivalent: the generator must be monotone in alpha. |
| `test_ppc_locks_phases` | Ppc locks phases. |
| `test_aac_correlates_envelopes` | Aac correlates envelopes. |
| `test_nonstationary_profiles` | Nonstationary profiles. |
| `test_reproducible_from_seed` | Reproducible from seed. |
| `test_nyquist_violation_is_caught` | Nyquist violation is caught. |
| `test_parallel_rngs_are_independent` | Results must not depend on n_jobs. |
| `test_ppc_plv_rises_monotonically_with_alpha` | The bug this covers: an earlier version mixed unwrapped phases convexly, which produced a weighted-average frequency rather than locking. PLV was ~0 at every alpha except exactly 1. |
| `test_ppc_locked_component_stays_in_its_band` | The locked oscillation must survive a band-pass at the high band. |
| `test_incoherent_nm_ratio_warns` | Incoherent nm ratio warns. |
| `test_harmonic_contamination_actually_fakes_coupling` | The bug this covers: the artefact was advertised for several phases and produced nothing, because the distortion was applied to the waveform of a band-limited noise rather than to its phase [DD-64]... |
| `test_harmonic_artefact_is_monotone_in_contamination` | Harmonic artefact is monotone in contamination. |
| `test_harmonic_artefact_makes_the_slow_rhythm_non_sinusoidal` | The mechanism, measured directly: harmonics of the slow frequency. |
| `test_sharpness_controls_how_far_the_harmonics_reach` | Sharpness controls how far the harmonics reach. |
| `test_truth_records_the_sharpness` | Truth records the sharpness. |
| `test_a_scatter_cannot_show_coupling_but_the_mean_can` | DD-106. The spread of amplitude within one phase bin is several times the movement of the mean across bins, so individual points drown the effect even when the coupling is strong. Averaging recover... |
| `test_the_modulogram_recovers_the_preferred_phase` | The modulogram recovers the preferred phase. |
| `test_the_modulogram_is_flat_without_coupling` | The modulogram is flat without coupling. |
| `test_the_modulogram_reports_its_own_uncertainty` | A bin with few samples wanders on its own; a peak inside the error bars is not a peak. |
| `test_the_comodulogram_finds_which_pair_couples` | The map that says which band pair couples rather than assuming one. |
| `test_coupling_indices_carry_the_modulogram_shape` | Coupling indices carry the modulogram shape. |
| `test_a_comodulogram_needs_both_roles` | A comodulogram needs both roles. |
| `test_surrogates_are_reachable_from_the_top_level` | It existed from the first phase and was only ever reachable as recurra.metrics.classic.surrogate_significance, which is where a user running a study will not look. The omission cost a full 31-chann... |
| `test_surrogates_separate_real_coupling_from_its_own_null` | The baseline for 'no coupling' cannot be a synthetic number: real signals are 1/f with non-sinusoidal waveforms and both raise the modulation index without any coupling. Measured, the surrogate nul... |
| `test_the_surrogate_z_score_saturates_and_the_p_value_does_not` | DD-107. A circular shift preserves the envelope entirely, so an envelope genuinely modulated at the phase frequency stays modulated after shifting -- only its preferred phase moves. The null theref... |
| `test_modulogram_wraps_the_phase_like_the_modulation_index` | A phase given in [0, 2*pi) must bin as the same phase in (-pi, pi]. |
| `test_canonical_indices_flag_an_amplitude_that_is_not_an_envelope` | MVL and MI are defined on an envelope. A negative 'amplitude' warns, and MI refuses once a phase bin's mean amplitude is negative, because the divergence is then taken from something that is not a ... |

## 20. `test_statespace.py` — State space construction

The weight-folding identity, every constructor on all three routes, embedding estimation, scaling and transforms.

Decisions: DD-21, DD-23, DD-25, DD-28

| Test | What it checks |
|---|---|
| `test_weighted_coords_reproduce_grouped_distance` | DD-21: Euclidean distance on weighted coords IS the grouped distance. |
| `test_coords_are_read_only` | Coords are read only. |
| `test_groups_partition_the_columns` | Groups partition the columns. |
| `test_phase_circle_lies_on_unit_circle` | Phase circle lies on unit circle. |
| `test_pac_space_shape_and_blocks` | Pac space shape and blocks. |
| `test_pac_space_picks_slow_phase_and_fast_amplitude` | Pac space picks slow phase and fast amplitude. |
| `test_pac_space_refuses_when_bands_unknown` | Pac space refuses when bands unknown. |
| `test_observable_builders` _(×4)_ | Observable builders. |
| `test_ppc_space_has_two_circles` | Ppc space has two circles. |
| `test_ppa_space_is_five_dimensional` | Ppa space is five dimensional. |
| `test_channels_needs_no_coupling` | Nothing forces the CFC reading. |
| `test_custom_callable` | Custom callable. |
| `test_takens_shape` | Takens shape. |
| `test_takens_auto_recovers_lorenz_dimension` | AMI + normalised FNN should find m=3 for Lorenz [DD-23]. |
| `test_takens_multivariate` | Takens multivariate. |
| `test_impossible_embedding_is_rejected` | Impossible embedding is rejected. |
| `test_hybrid_keeps_circle_and_embeds_envelope` | Hybrid keeps circle and embeds envelope. |
| `test_blocks_of_different_length_are_aligned_and_flagged` | Blocks of different length are aligned and flagged. |
| `test_from_bands_end_to_end` | From bands end to end. |
| `test_default_scaling_balances_blocks` | Default scaling balances blocks. |
| `test_scaling_none_leaves_imbalance_visible` | Scaling none leaves imbalance visible. |
| `test_reweight_sets_share_to_lambda` _(×5)_ | Reweight sets share to lambda. |
| `test_tail_ratio_of_unit_circle_is_sqrt_two` | DD-28: reference value for the shape diagnostic. |
| `test_heavy_tailed_block_is_flagged` | Heavy tailed block is flagged. |
| `test_smoothing_is_recorded` | Smoothing is recorded. |
| `test_subsample_records_caveat` | Subsample records caveat. |
| `test_reduce_pca` | Reduce pca. |
| `test_split_extracts_blocks` | Split extracts blocks. |
| `test_window_shifts_t0` | Window shifts t0. |
| `test_compare_routes_returns_row_per_route` | Compare routes returns row per route. |
| `test_lambda_sweep_needs_two_blocks` | Lambda sweep needs two blocks. |
| `test_lambda_sweep_endpoints_are_pure` | Lambda sweep endpoints are pure. |
| `test_neighbour_agreement_is_one_for_identical` | Neighbour agreement is one for identical. |
| `test_plot_modes_render` | Plot modes render. |
| `test_torus_mode_needs_a_phase_circle` | Torus mode needs a phase circle. |
| `test_save_attractor_writes` | Save attractor writes. |
| `test_lambda_sweep_endpoints_are_uninformative_about_coupling` | Both ends of the sweep are single-block projections, so a coupled and an uncoupled signal must agree there [DD-02]. |
| `test_coupling_lowers_the_joint_dimension` | The corrected sign of DD-02: coupling is a constraint, and a constraint removes dimension. The uncoupled signal, whose blocks form a cartesian product, must show the *larger* interior slope. |
| `test_lambda_sweep_reports_a_deficit_not_a_synergy` | Lambda sweep reports a deficit not a synergy. |
| `test_shape_descriptors_report_the_tail_per_block` | DD-102. The warning existed from the start and nothing recorded how heavy the tail was, so nobody could check whether it differed between the groups being compared. |
| `test_the_tail_comes_free_with_the_geometry` | It is the quantity most likely to confound a group comparison, so it should not need asking for [DD-102]. |
| `test_pac_space_warns_when_the_amplitude_band_cannot_hold_the_sidebands` | A 10 Hz band around 40 Hz cannot carry an 8 Hz modulation. |
| `test_pac_space_warns_when_smoothing_cuts_the_phase_band` | An envelope low-pass below the phase band removes the coupling. |
| `test_gaussian_smoothing_keeps_the_envelope_non_negative` | DD-117. The Butterworth smoother rang below zero on a strongly modulated envelope (4.7% of samples at alpha 0.9). A non-negative kernel cannot. |
| `test_both_smoothers_halve_the_amplitude_at_the_cut_off` _(×2)_ | The cut-off means the same thing for both methods: gain 1/2. |
| `test_unknown_smoothing_method_is_refused` | Unknown smoothing method is refused. |
| `test_ami_matches_an_independent_histogram_estimate` | Average mutual information against a plain 2-D histogram computation. |
| `test_lorenz_embedding_is_recovered` | Ground truth: tau about 16 samples at dt=0.01 by AMI, and m=3 by FNN. |
| `test_fnn_handles_an_exactly_periodic_series` | A sine sampled at an integer number of samples per period repeats its points to within rounding. FNN used to count those duplicates as false neighbours and return the top of m_range (12); a circle ... |
| `test_an_estimate_at_the_edge_of_its_range_warns` | P3: a value forced by the search range, not by the data, says so. |

## 21. `test_threshold.py` — Threshold selection

That every mode runs, that the target rate is met, and that Theiler-aware sampling behaves.

Decisions: DD-32, DD-35

| Test | What it checks |
|---|---|
| `test_sampler_honours_theiler` | DD-32: sampled pairs must come from the region RR is defined over. |
| `test_sampler_refuses_impossible_theiler` | Sampler refuses impossible theiler. |
| `test_target_rr_hits_the_target` | Target rr hits the target. |
| `test_bisect_converges_and_reports_iterations` | Bisect converges and reports iterations. |
| `test_fixed_reports_what_it_achieves` | Fixed reports what it achieves. |
| `test_percentile_rejects_out_of_range` | Percentile rejects out of range. |
| `test_fan_is_pointwise` | Fan is pointwise. |
| `test_fan_neighbour_count_sets_the_rate` | Fan neighbour count sets the rate. |
| `test_fan_never_counts_a_point_as_its_own_neighbour` _(×2)_ | DD-115: neighbours need \|i - j\| >= max(theiler, 1), so with theiler=0 the nearest neighbour is still another point, never the point itself. |
| `test_per_block_gives_one_epsilon_per_block` | Per block gives one epsilon per block. |
| `test_per_block_needs_a_statespace` | Per block needs a statespace. |
| `test_std_fraction_scales_with_the_attractor` | Std fraction scales with the attractor. |
| `test_adaptive_dim_warns_when_epsilon_swallows_the_attractor` | Adaptive dim warns when epsilon swallows the attractor. |
| `test_unknown_mode_is_rejected` | Unknown mode is rejected. |
| `test_all_documented_modes_run` | All documented modes run. |
| `test_threshold_frame_is_exportable` | Threshold frame is exportable. |
| `test_cross_threshold_requires_matching_dimensions` | Cross threshold requires matching dimensions. |
| `test_cross_threshold_hits_target` | Cross threshold hits target. |

## 22. `test_validation.py` — test_validation.py

| Test | What it checks |
|---|---|
| `test_every_signal_has_its_own_seed` | Guarding the design flaw that produced a false null [DD-65]. |
| `test_the_artefact_covers_the_same_classical_range` | Without overlap the comparison is about coupling strength, not mechanism. |
| `test_determinism_distinguishes_genuine_coupling_from_the_artefact` | DD-65. At equal modulation index, DET is higher for genuine coupling. The effect is modest -- about 0.006 on a scale where DET is 0.93 -- so this test uses an uncorrected threshold and a directiona... |
| `test_the_vertical_family_points_the_other_way` | Trapping time is *higher* for the artefact, opposite to the diagonal family. A mechanism that flipped the sign of one without the other would mean something changed [DD-65]. |
| `test_classical_index_responds_across_the_sweep` | The premise: MI varies within each mechanism, which is why matching on it is necessary and why the artefact is a problem worth testing. |
| `test_classical_indices_detect_weaker_coupling_than_det` | DD-66. The framework must not be sold on sensitivity: a modulation index detects coupling that DET misses, at a hundredth of the computation. Measured on the full surface: MI reaches AUC 0.90 at al... |
| `test_strong_coupling_is_detected_by_everything` | The other end of the surface: at alpha 0.9 there is nothing subtle left. |
| `test_the_vertical_family_detects_better_than_det` | DD-66: LAM reaches a floor of 0.50 where DET needs 0.90, inverting the usual emphasis on determinism. |
| `test_these_metrics_do_not_drift_with_window_length` _(×3)_ | DD-67: RR, DET and LAM are the only ones stable enough to compare between records of different length. |
| `test_lmax_scales_with_the_window` | DD-67: Lmax is bounded by N, so it grows with the window and cannot be compared between records of different length at all. |
| `test_normalised_entropy_does_not_fix_the_length_dependence` | DD-59, corrected: dividing by log(distinct lengths) overcorrects, so ENTR_norm drifts *more* than raw ENTR and in the opposite direction. |
| `test_one_second_windows_detect_coupling` | DD-67: trapping time separates coupled from control at 500 points, which sets the temporal resolution of the method at about one cycle of the modulating rhythm rather than tens. |
| `test_lam_works_in_a_shorter_window_than_det` | DD-67. LAM separates coupled from control at two seconds; DET needs sixteen. A four-second window -- a common default -- is below the length at which DET functions. |
| `test_det_lam_and_l_are_length_invariant` | DD-67: the ratio family barely moves with window length, which is what lets it be compared across records once N is matched [DD-43]. |
| `test_lmax_is_not_length_invariant` | The counter-case: Lmax is the longest line found, so it can only grow with the record and must never be compared across lengths [DD-67]. |
| `test_a_single_short_window_is_not_enough` | DD-69, and DD-68 by a different route: one window needs about 4 s -- 24 cycles of the slow rhythm -- before any measure classifies a subject. Half a second does not. |
| `test_averaging_many_short_windows_recovers_what_one_cannot` | DD-69: with the whole record available, cutting finely and averaging beats using few long windows. The two analyses answer different questions and must not be conflated. |
| `test_recurrence_beats_plv_on_phase_phase_coupling` | DD-70. The first case where a recurrence measure clearly wins. Band-pass filtering at the high band mixes the locked component with the free-running one, so the measured PLV collapses -- from about... |
| `test_only_recurrence_sees_trivariate_coupling` | DD-70. In PPA the fast envelope follows the *sum* of two slow phases, so a bivariate index between one of them and the envelope sees nothing. This is the clearest argument for the framework: a coup... |
| `test_hjorth_activity_is_a_scale_confound_not_a_finding` | DD-74. Hjorth activity is the variance of the series, and the harmonic artefact adds energy in the high band by construction. It separates the two conditions strongly and means nothing dynamical. T... |
| `test_higuchi_of_the_envelope_is_scale_invariant` | DD-74: the reason Higuchi is the defensible discriminator and Hjorth activity is not. Scaling the series must not move the dimension. |
| `test_phase_block_measures_are_blind_to_pac` | DD-75. PAC modulates the amplitude of the fast rhythm, not the dynamics of the slow phase, so no measure of the phase block should detect it. A measure that did would be a warning, not a result. |
| `test_envelope_measures_do_detect_pac` | The other half of the same check: the amplitude block does see it. |
| `test_svd_entropy_beats_rqa_on_a_one_second_window` | DD-76. A single window needs 2-4 s for RQA and 1 s for SVD entropy: a factor of four in the temporal resolution a time-resolved analysis can claim, from a measure outside the recurrence family. |
| `test_every_measure_is_computable_on_a_short_window` | A measure that quietly fails at 250 points would read as one that detects nothing [DD-76]. |
| `test_a_univariate_measure_cannot_tell_coupling_from_regularity` | DD-77. Nine block-wise measures separated PPC at AUC 1.000, beating PLV. They are computed on the gamma block alone and never see theta. Pairing a regular gamma phase with an *independent* theta mu... |

## 23. `test_viz.py` — Figures

That every registered figure renders from one object, that inputs are not mutated, and the language guards.

Decisions: DD-27, DD-30, DD-31, DD-38, DD-49..52

| Test | What it checks |
|---|---|
| `test_attractor_modes` _(×6)_ | Attractor modes. |
| `test_every_single_figure_takes_one_object` | Each entry of the registry must render from a single object. |
| `test_single_figure_does_not_mutate_its_input` | Single figure does not mutate its input. |
| `test_plot_functions_return_without_saving` | Plot functions return a Figure; saving is a separate, explicit step. |
| `test_comparison_takes_a_mapping` | Comparison takes a mapping. |
| `test_comparison_rejects_empty_mapping` | Comparison rejects empty mapping. |
| `test_comparison_shares_limits` | Shared axes are what makes a comparison honest. |
| `test_registries_are_complete` | Registries are complete. |
| `test_figure_text_is_english` | Titles, axis labels and legends carry no accented (Spanish) text. |
| `test_dataframe_columns_are_english` | Dataframe columns are english. |
| `test_source_has_no_spanish_identifiers` | The package source is English-only [DD-30]. |
| `test_max_pooling_saturation_is_predicted` | Max pooling saturation is predicted. |
| `test_auto_pooling_avoids_saturation` | Auto pooling avoids saturation. |
| `test_pooled_image_is_not_saturated` | The bug this rule exists for: a 5x reduction turning the plot solid. |
| `test_render_streams_identically_to_dense` _(×3)_ | Bin edges must be anchored to the global grid, not to tile boundaries. |
| `test_shared_limits_refused_for_incomparable_spaces` | DD-49: a Lorenz attractor shares axes with a PAC space only by shrinking the latter to a dot. |
| `test_auto_share_limits_leaves_panels_independent` | Auto share limits leaves panels independent. |
| `test_scale_report_has_shape_stats_without_a_policy` | DD-28 diagnostics are wanted most where no policy was applied. |
| `test_scale_report_figure_draws_both_panels` | Scale report figure draws both panels. |
| `test_robust_limits_ignore_rare_extremes` | DD-50: instantaneous frequency spikes must not squash everything else. |
| `test_quality_panel_lists_the_checks` | DD-52: a panel saying only 'no issues' reads as one that failed. |
| `test_quality_panel_marks_failures` | Quality panel marks failures. |
| `test_pairs_diagonal_shows_phase_for_circle_blocks` | DD-51: the histogram of cos is an arcsine U-shape that says nothing. |
| `test_route_figure_excludes_scale_dependent_metrics_by_default` | nn_mean is in the units of each space, so routes with different scaling policies cannot be compared on it [DD-53]. |
| `test_route_figure_flags_a_scale_dependent_metric_when_asked` | Route figure flags a scale dependent metric when asked. |
| `test_route_figure_reports_failed_routes` | A route that did not apply must not vanish from the figure. |
| `test_routes_do_not_see_the_same_geometry` | DD-53: the choice of route is not an implementation detail. |
| `test_shared_limits_cover_every_space` | DD-104. Two attractors on their own axes are two pictures of two shapes: the reader cannot tell a wider trajectory from a different scale. |
| `test_plot_attractor_honours_explicit_limits` | Plot attractor honours explicit limits. |
| `test_every_comparison_figure_is_reachable_from_the_top_level` | DD-104: they existed only as rc.viz.compare.something while every single-object figure was rc.something. Nothing intended that. |
| `test_the_weighted_scale_follows_the_recordings_gain` | DD-105. rms_balanced fixes the *ratio* between blocks and leaves the absolute scale at n / sum(1/rms^2), which follows the raw amplitude. On a real corpus epsilon ranged over a factor of eight betw... |
| `test_normalising_makes_the_same_shape_look_the_same` | Normalising makes the same shape look the same. |
| `test_shared_limits_normalise_by_default` | Shared limits normalise by default. |
| `test_the_figure_says_when_it_has_normalised` | The figure says when it has normalised. |
| `test_the_recurrence_plot_never_used_the_absolute_scale` | The normalisation is a display decision only: epsilon follows the scale and the recurrence rate is pinned, so nothing computed changes [DD-105]. |
| `test_the_polar_view_puts_phase_on_the_angle` | The view that makes coupling legible by eye: a coupled recording is an eccentric ring, an uncoupled one a round band. |
| `test_the_panel_draws_every_kind_of_plot` | RP, CRP, JRP and meta-RP all render with their signals in the margins. |
| `test_the_panel_accepts_raw_signals_for_its_margins` | The panel accepts raw signals for its margins. |
| `test_the_joint_figure_refuses_a_plain_plot` | The joint figure refuses a plain plot. |

## 24. `test_windowed.py` — Windowed analysis

Window specification, threshold scope, laziness, every mode windowed, and the multi-record batch.

Decisions: DD-39, DD-40, DD-41

| Test | What it checks |
|---|---|
| `test_n_windows_gives_exactly_that_many` | Asking for 10 must give 10, not 9 with a tail left over. |
| `test_n_windows_spans_the_whole_record` | N windows spans the whole record. |
| `test_overlap_is_realised` _(×4)_ | Overlap is realised. |
| `test_size_and_step_in_seconds` | Size and step in seconds. |
| `test_size_and_overlap` | Size and overlap. |
| `test_unit_samples` | Unit samples. |
| `test_uncovered_tail_is_reported` | Uncovered tail is reported. |
| `test_drop_last_false_pulls_the_window_back` | Drop last false pulls the window back. |
| `test_overlap_as_percentage_is_rejected` | Overlap as percentage is rejected. |
| `test_step_and_overlap_together_rejected` | Step and overlap together rejected. |
| `test_window_larger_than_record_is_rejected` | Window larger than record is rejected. |
| `test_resolve_accepts_int_and_explicit_pairs` | Resolve accepts int and explicit pairs. |
| `test_window_times_are_right` | Window times are right. |
| `test_ten_windows_ten_matrices` | The headline case: one 60 s record, ten overlapping recurrence plots. |
| `test_per_window_scope_fixes_the_rate` | Per window scope fixes the rate. |
| `test_global_scope_fixes_epsilon_and_frees_the_rate` | DD-40: the scope decides what a windowed analysis can say. |
| `test_scope_must_be_valid` | Scope must be valid. |
| `test_windows_are_lazy_by_default` | Windows are lazy by default. |
| `test_keep_caches_the_matrices` | Keep caches the matrices. |
| `test_windows_are_the_right_segments` | Windows are the right segments. |
| `test_metrics_frame_is_tidy_and_complete` | Metrics frame is tidy and complete. |
| `test_series_extracts_one_metric` | Series extracts one metric. |
| `test_all_threshold_modes_window` _(×7)_ | All threshold modes window. |
| `test_all_distance_metrics_window` _(×3)_ | All distance metrics window. |
| `test_windowed_cross_recurrence` | Windowed cross recurrence. |
| `test_windowed_joint_recurrence_keeps_independent_thresholds` | Windowed joint recurrence keeps independent thresholds. |
| `test_jrp_windows_detect_coupling` | Coupled windows should show an independence ratio above 1. |
| `test_crp_needs_exactly_two` | Crp needs exactly two. |
| `test_misaligned_trajectories_rejected` | Misaligned trajectories rejected. |
| `test_takens_route_windows` | Takens route windows. |
| `test_raw_array_windows` | Raw array windows. |
| `test_hybrid_space_windows` | Hybrid space windows. |
| `test_batch_over_subjects` | Batch over subjects. |
| `test_batch_handles_unequal_lengths` | Real corpora have records of different length [DD-18]. |
| `test_batch_results_do_not_depend_on_order` | Batch results do not depend on order. |
| `test_batch_from_a_sequence` | Batch from a sequence. |
| `test_provenance_carries_the_window_spec` | Provenance carries the window spec. |
| `test_describe_summarises` | Describe summarises. |
| `test_window_figures_render` | Window figures render. |
| `test_windows_feed_the_comparison_figures` | Windows feed the comparison figures. |
| `test_independence_ratio_detects_a_transient_at_the_right_time` | Ground truth: coupling between 19.8 s and 39.6 s. The detected onset and offset must fall within one window step of those [DD-54]. |
| `test_subsystem_rates_do_not_reveal_the_transient` | The point of the joint construction: neither subsystem changes [DD-54]. |
| `test_classical_indices_enter_the_windowed_table` | DD-106. A study can compute recurrence quantification of a phase-amplitude space for months without establishing that the space contains any coupling. The reference belongs in the same table. |
| `test_the_sweep_reuses_the_histograms_instead_of_re_walking` | DD-108. Every RQA metric at every line-length floor is a sum over the line histogram, so exploring floors should be free. It was not: calling metrics() again with a different floor invalidates its ... |
| `test_sweeping_before_any_rqa_says_so` | Sweeping before any rqa says so. |
| `test_batch_metrics_parallelise_and_agree` | DD-109. The build was parallel and the metrics were not, though they are four fifths of the work. Results must not depend on n_jobs, and the histograms DD-108 caches must survive the process bounda... |
| `test_windowed_metrics_share_the_rqa_floors` | The two entry points must default to the same l_min and v_min. |
| `test_classical_indices_refuse_a_rescaled_amplitude` | A z-scored amplitude is not an envelope: computed on it, MVL came out at 13.3 and MI at 0.88 on a signal whose values were 0.31 and 0.035, with no warning. The table now says NaN and the reason, once. |
| `test_every_explicit_option_reaches_the_window_thresholds` _(×2)_ | Regression [DD-118]: the options must reach the threshold estimator, not only the plot. In the first 0.25 draft they arrived after the thresholds had been estimated, so target_rr=0.20, theiler=7 an... |

"""Validation-only event threshold calibration of existing Deform scores (CPU)."""
from .common import project_path, relative_path
import argparse
import datetime
import json
import shutil
from pathlib import Path

import numpy as np

from .common import ROOT, dump, sha
from .early_warning import HORIZONS, SEEDS, metrics, read_data, target, write_csv

LIMITS = (.10, .15, .20)


def binary_metrics(y, score, threshold):
    # Preserve original score ranking for AP/ROC; only discrete metrics change.
    result = metrics(y, score)
    y = np.asarray(y, int)
    prediction = np.asarray(score, dtype=np.float64) >= threshold
    cm = np.array([[int(((y == a) & (prediction == b)).sum())
                    for b in (0, 1)] for a in (0, 1)])
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(2), where=cm.sum(1) > 0)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(2), where=cm.sum(0) > 0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(2), where=precision+recall > 0)
    result.update(accuracy=float((prediction == y).mean()), balanced_accuracy=float(recall.mean()),
                  precision=float(precision[1]), recall=float(recall[1]), f1=float(f1[1]),
                  macro_f1=float(f1.mean()), fpr=float(1-recall[0]), confusion_matrix=cm.tolist())
    return result


def event_scores(samples, p, offsets, h):
    gt, scores = [], []
    for i, s in enumerate(samples):
        _, valid = target(s['frames'], s['key'], s['outcome'] == 2, h)
        local = p[offsets[i]:offsets[i+1]].astype(np.float64)
        assert len(local) == len(valid) and valid.any()
        gt.append(int(s['outcome'] == 2))
        scores.append(float(local[valid].max()))
    return np.array(gt), np.array(scores)


def select_threshold(y, scores, fpr_limit):
    # Include a feasible all-negative candidate, including the exact score=1 case.
    candidates = np.unique(np.r_[scores, 1., np.nextafter(max(1., float(scores.max())), np.inf)])
    negative = int((y == 0).sum())
    allowed = int(np.floor(fpr_limit*negative + 1e-12))
    choices = []
    for threshold in candidates:
        prediction = scores >= threshold
        fp = int(((y == 0) & prediction).sum())
        tp = int(((y == 1) & prediction).sum())
        if fp <= allowed:
            choices.append((tp, float(threshold), fp))
    tp, threshold, fp = max(choices, key=lambda v: (v[0], v[1]))
    assert fp/negative <= fpr_limit + 1e-12
    return dict(threshold=threshold, allowed_success_fp=allowed, selected_success_fp=fp,
                selected_failure_tp=tp, candidate_count=len(candidates),
                validation_success=negative, validation_failure=int((y == 1).sum()))


def evaluate(samples, p, offsets, h, seed, split, threshold, limit):
    yy, pp, rows = [], [], []
    for i, s in enumerate(samples):
        frames = s['frames']
        local = p[offsets[i]:offsets[i+1]].astype(np.float64)
        failure = s['outcome'] == 2
        y, valid = target(frames, s['key'], failure, h)
        yy.append(y[valid]); pp.append(local[valid])
        hit = np.flatnonzero(valid & (local >= threshold))
        first = int(hit[0]) if len(hit) else None
        first_in_band = bool(first is not None and failure and y[first] == 1)
        lead = (s['key']-int(frames[first]))/30 if first is not None else None
        rows.append(dict(horizon_frames=h, seed=seed, fpr_limit=limit, split=split,
                         threshold=threshold, rollout_id=s['rollout_id'], final_failure=int(failure),
                         event_score=float(local[valid].max()), alarm=first is not None,
                         first_alarm_frame=int(frames[first]) if first is not None else None,
                         anchor=int(s['key']), first_alarm_lead_seconds=lead,
                         positive_band_detected=bool(np.any(valid & (y == 1) & (local >= threshold))),
                         first_alarm_in_positive_band=first_in_band,
                         effective_early_warning_lead_seconds=lead if first_in_band and h > 0 else None))
    ey, es = event_scores(samples, p, offsets, h)
    event = binary_metrics(ey, es, threshold)
    failures = [r for r in rows if r['final_failure']]
    all_leads = [r['first_alarm_lead_seconds'] for r in failures if r['alarm']]
    correct_leads = [r['effective_early_warning_lead_seconds'] for r in failures
                     if r['effective_early_warning_lead_seconds'] is not None]
    event.update(positive_band_detection_rate=sum(r['positive_band_detected'] for r in failures)/len(failures),
                 correct_first_alarm_rate=sum(r['first_alarm_in_positive_band'] for r in failures)/len(failures),
                 detected_failure_count=len(all_leads), correct_early_warning_count=len(correct_leads),
                 failure_count=len(failures), undecided_rate=sum(not r['alarm'] for r in rows)/len(rows),
                 median_first_alarm_lead_seconds=float(np.median(all_leads)) if all_leads else None,
                 median_effective_early_warning_lead_seconds=float(np.median(correct_leads)) if correct_leads else None)
    return binary_metrics(np.concatenate(yy), np.concatenate(pp), threshold), event, rows


def mean_std(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None, len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=project_path('outputs/sharpa_early_warning/20261006_001500'))
    parser.add_argument('--features', type=Path, default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    samples = read_data(args.features.resolve(), False)
    originals = json.loads((source/'results.json').read_text())
    assert len(originals) == 25
    results, event_rows, flat_rows, baseline, provenance = [], [], [], [], []
    curve_cache = {}
    for h in HORIZONS:
        for seed in SEEDS:
            directory = source/'runs'/f'H{h}'/f'seed_{seed}'
            arrays = {}
            original = next(r for r in originals if r['horizon_frames'] == h and r['seed'] == seed)
            for split in ('val', 'test'):
                path = directory/(split+'_predictions.npz')
                with np.load(path) as z:
                    p, offsets = z['probabilities'].copy(), z['offsets'].copy()
                assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                assert np.array_equal(offsets, np.r_[0, np.cumsum([len(s['frames']) for s in samples[split]])])
                arrays[split] = (p, offsets)
                provenance.append(dict(path=str(relative_path(path)), sha256=sha(path)))
                f, e, _ = evaluate(samples[split], p, offsets, h, seed, split, .5, None)
                assert f['confusion_matrix'] == original[split]['confusion_matrix']
                assert e['confusion_matrix'] == original[split+'_event']['confusion_matrix']
                for phase, recomputed in ((split, f), (split+'_event', e)):
                    for key in ('pr_auc', 'roc_auc', 'balanced_accuracy'):
                        assert np.isclose(recomputed[key], original[phase][key])
                baseline.append(dict(horizon_frames=h, seed=seed, split=split, frame=f, event=e))
            vy, vs = event_scores(samples['val'], *arrays['val'], h)
            curve_cache[(h, seed)] = arrays
            for limit in LIMITS:
                selection = select_threshold(vy, vs, limit)
                threshold = selection['threshold']
                selection['alarm_disabled'] = threshold > 1.
                result = dict(horizon_frames=h, seed=seed, fpr_limit=limit, **selection)
                for split in ('val', 'test'):
                    frame, event, rows = evaluate(samples[split], *arrays[split], h, seed, split, threshold, limit)
                    # Explicitly recompute event TP/FP directly from saved scores and threshold.
                    eg, score = event_scores(samples[split], *arrays[split], h)
                    alarm = score >= threshold
                    cm = [[int(((eg == gt) & (alarm == pred)).sum()) for pred in (0, 1)] for gt in (0, 1)]
                    assert cm == event['confusion_matrix']
                    if split == 'val':
                        assert cm[0][1] == selection['selected_success_fp'] <= selection['allowed_success_fp']
                        assert cm[1][1] == selection['selected_failure_tp']
                    assert np.isclose(frame['pr_auc'], original[split]['pr_auc'])
                    assert np.isclose(event['roc_auc'], original[split+'_event']['roc_auc'])
                    result[split], result[split+'_event'] = frame, event
                    event_rows.extend(rows)
                    flat = dict(horizon_frames=h, seed=seed, fpr_limit=limit, threshold=threshold, split=split,
                                validation_allowed_success_fp=selection['allowed_success_fp'])
                    for phase, metric in (('frame', frame), ('event', event)):
                        for key, value in metric.items():
                            if key != 'confusion_matrix':
                                flat[phase+'_'+key] = value
                    flat_rows.append(flat)
                results.append(result)
            print('CALIBRATED', h, seed, flush=True)
    assert len(results) == 75
    assert {(r['horizon_frames'], r['seed'], r['fpr_limit']) for r in results} == {
        (h, seed, limit) for h in HORIZONS for seed in SEEDS for limit in LIMITS}
    for h in HORIZONS:
        for seed in SEEDS:
            selected = {r['fpr_limit']: r['threshold'] for r in results
                        if r['horizon_frames'] == h and r['seed'] == seed}
            assert selected[.15] == selected[.20]
    dump(out/'results.json', results)
    write_csv(out/'calibration_thresholds.csv', [
        {key: value for key, value in result.items()
         if key not in ('val', 'test', 'val_event', 'test_event')}
        for result in results])
    dump(out/'baseline_metrics.json', baseline)
    dump(out/'score_provenance.json', provenance)
    write_csv(out/'thresholds_and_metrics.csv', flat_rows)
    write_csv(out/'event_predictions.csv', event_rows)
    summaries, matrices = [], []
    for h in HORIZONS:
        for limit in LIMITS:
            chosen = [r for r in results if r['horizon_frames'] == h and r['fpr_limit'] == limit]
            summary = dict(horizon_frames=h, horizon_seconds=h/30, fpr_limit=limit, seeds=5)
            for key in ('threshold',):
                summary[key+'_mean'], summary[key+'_std'], summary[key+'_valid_seeds'] = mean_std([r[key] for r in chosen])
            for phase in ('val', 'test', 'val_event', 'test_event'):
                for key in chosen[0][phase]:
                    if key == 'confusion_matrix':
                        continue
                    mean, sd, n = mean_std([r[phase][key] for r in chosen])
                    summary[phase+'_'+key+'_mean'] = mean
                    summary[phase+'_'+key+'_std'] = sd
                    summary[phase+'_'+key+'_valid_seeds'] = n
                for gt in (0, 1):
                    total = sum(sum(r[phase]['confusion_matrix'][gt]) for r in chosen)
                    for pred in (0, 1):
                        count = sum(r[phase]['confusion_matrix'][gt][pred] for r in chosen)
                        matrices.append(dict(horizon_frames=h, fpr_limit=limit, phase=phase,
                                             ground_truth=gt, prediction=pred, mean_count=count/5,
                                             row_normalized=count/total))
                split = phase.replace('_event', '')
                originals_for_phase = [r for r in baseline if r['horizon_frames'] == h and r['split'] == split]
                base_key = 'event' if phase.endswith('_event') else 'frame'
                for key in ('precision', 'recall', 'fpr', 'positive_band_detection_rate', 'correct_first_alarm_rate'):
                    if key in originals_for_phase[0][base_key]:
                        mean, sd, _ = mean_std([r[base_key][key] for r in originals_for_phase])
                        summary['baseline_'+phase+'_'+key+'_mean'] = mean
                        summary['baseline_'+phase+'_'+key+'_std'] = sd
            summaries.append(summary)
    write_csv(out/'summary.csv', summaries)
    write_csv(out/'confusion_normalized.csv', matrices)
    plot_curves(out, samples, curve_cache, results)
    write_readme(out, source, summaries)
    dump(out/'verification.json', dict(status='PASS', calibration_runs=75, pretrained_runs=25,
         calibration_uses_validation_only=True, fixed_test_thresholds=True,
         validation_fp_constraints_checked=True, event_confusions_recomputed=True,
         source_baseline_frame_event_metrics_reproduced=True, original_auc_unchanged=True,
         comparison_float64=True, sensitivity_15_20_identical=True,
         same_rollout_split=True, no_training=True, no_checkpoint_or_score_modifications=True))
    dump(out/'protocol.json', dict(created_at=datetime.datetime.now().astimezone().isoformat(),
         source=str(relative_path(source)), horizons=list(HORIZONS), seeds=list(SEEDS),
         fpr_limits=list(LIMITS), event_score='max risk over original horizon-valid observation range',
         alarm_rule='risk >= threshold', primary_fpr_limit=.15,
         selection='maximize validation failure event TP subject to success FP constraint; ties highest threshold',
         candidates='distinct validation event scores, 1.0, nextafter(max(1,max_score), +inf)',
         zero_recall='highest feasible candidate above all event scores: no alarm; no arbitrary fallback',
         terminal_mask='H>0 failure frame<anchor; success all frames; H0 all frames',
         no_training=True, cpu_only=True, code_sha256=sha(Path(__file__))))
    snap = out/'code_snapshot'; snap.mkdir()
    shutil.copy2(Path(__file__), snap/Path(__file__).name)
    destination = project_path('WeeklySummary/10.5/early_warning_calibration', out.name)
    destination.mkdir(parents=True, exist_ok=False)
    copies = []
    for path in out.iterdir():
        if path.is_file():
            copied = destination/path.name; shutil.copy2(path, copied)
            digest = sha(path); assert sha(copied) == digest
            copies.append(dict(source=str(relative_path(path)), destination=str(relative_path(copied)), sha256=digest))
    dump(destination/'copy_manifest.json', copies)
    print('CALIBRATION_COMPLETE', json.dumps([r for r in summaries if r['fpr_limit'] == .15]), flush=True)


def plot_curves(out, samples, cache, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve_rows = []
    for split in ('val', 'test'):
        riskfig, riskaxes = plt.subplots(2, 5, figsize=(23, 8))
        marginfig, marginaxes = plt.subplots(2, 5, figsize=(23, 8))
        for hi, h in enumerate(HORIZONS):
            thresholds = np.array([next(r['threshold'] for r in results if r['horizon_frames'] == h
                                       and r['seed'] == seed and r['fpr_limit'] == .15) for seed in SEEDS])
            ps = np.stack([cache[(h, seed)][split][0] for seed in SEEDS])
            offsets = cache[(h, SEEDS[0])][split][1]
            low = min(int(s['frames'].min()-s['key']) for s in samples[split])
            upper = max(int(s['frames'].max()-s['key']) for s in samples[split]) if h == 0 else -1
            grid = np.arange(low, upper+1)
            for oi, outcome in enumerate((1, 2)):
                scores, margins = [], []
                risk_mean = None
                for i, s in enumerate(samples[split]):
                    if s['outcome'] != outcome:
                        continue
                    rel = s['frames']-s['key']
                    local = ps[:, offsets[i]:offsets[i+1]]
                    score, margin = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
                    for frame in np.unique(rel[rel <= upper]):
                        score[int(frame-low)] = local[:, rel == frame].mean()
                        margin[int(frame-low)] = (local[:, rel == frame]-thresholds[:, None]).mean()
                    scores.append(score); margins.append(margin)
                for kind, matrix, axes in (('risk', scores, riskaxes), ('margin', margins, marginaxes)):
                    matrix = np.stack(matrix)
                    mean, sd = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
                    for j, frame in enumerate(grid):
                        finite = matrix[:, j][np.isfinite(matrix[:, j])]
                        if len(finite): mean[j] = finite.mean()
                        if len(finite) > 1: sd[j] = finite.std(ddof=1)
                        curve_rows.append(dict(split=split, horizon_frames=h, final_outcome=outcome,
                                               curve=kind, relative_seconds=frame/30, mean=float(mean[j]),
                                               variance=float(sd[j]**2), rollouts=len(finite),
                                               fpr_limit=.15, threshold_mean=float(thresholds.mean())))
                    ax = axes[oi, hi]; ax.plot(grid/30, mean, color='tab:red')
                    ax.fill_between(grid/30, mean-sd, mean+sd, color='tab:red', alpha=.15)
                    ax.axvline(0, color='black', ls=':')
                    if outcome == 2:
                        ax.axvspan(-h/30 if h else 0, 0 if h else upper/30, alpha=.10)
                    if kind == 'risk':
                        risk_mean = mean.copy()
                        ax.axhline(.5, color='gray', ls='--', label='original threshold=.5')
                        for index, threshold in enumerate(thresholds):
                            ax.axhline(threshold, color='tab:blue', alpha=.4, lw=.8,
                                       label='calibrated seed thresholds' if index == 0 else None)
                        ax.set_ylim(0, 1.02)
                    else:
                        ax.axhline(0, color='tab:blue', ls='--', label='calibrated decision boundary')
                        # Baseline and calibrated margins differ only by threshold, not score.
                        ax.plot(grid/30, risk_mean-.5, color='gray', ls=':',
                                label='original risk minus .5')
                        ax.set_ylim(-1.02, 1.02)
                    ax.set_title(f'H={h} | '+('success' if outcome == 1 else 'failure'))
                    ax.set_xlabel('Time relative to final Align end (s)')
                    ax.set_ylabel(kind+' mean ± rollout SD'); ax.legend(fontsize=6)
        for kind, fig in (('risk', riskfig), ('margin', marginfig)):
            fig.suptitle(split.upper()+' | validation event FPR constraint 15%; same cached scores')
            fig.tight_layout()
            fig.savefig(out/(split+'_key_relative_'+kind+'.png'), dpi=140)
            fig.savefig(out/(split+'_key_relative_'+kind+'.pdf')); plt.close(fig)
    write_csv(out/'key_relative_variance.csv', curve_rows)


def write_readme(out, source, summaries):
    def cell(row, key, percent=True):
        mean, sd = row[key+'_mean'], row[key+'_std']
        if mean is None: return '无有效首报'
        scale = 100 if percent else 1
        return f'{mean*scale:.2f} ± {sd*scale:.2f}' if sd is not None else f'{mean*scale:.2f}'
    lines = ['# Deform early warning：event-level threshold calibration', '',
        'Branch B：不重训、不改变checkpoint或缓存概率。对原实验25个H×seed分别只用validation校准阈值，随后固定到test。H={0,8,15,30,45}，seeds42–46；原Deform冻结encoder、balanced BCE、单向GRU128和rollout split均保持。', '',
        '## Anchor、范围与event score', '',
        '仅最后Align failure的end为anchor；早期Key与success Key不产生正标签。H>0的failure [anchor−H,anchor)为positive，anchor及之后不计训练/main evaluation；success全段negative。H0独立终态对照：failure anchor及之后为positive，观察全段。每个rollout取有效范围内最大risk作为event_score；risk≥threshold即alarm。GRU历史连续，未改输入或重算encoder。', '',
        'Val与test各17 rollout：14 success、3 failure。FPR=误报警success/14；recall=报警failure/3；precision=正确报警failure/全部报警rollout。FPR分辨率1/14=7.14%，recall分辨率1/3=33.33%。主约束val FPR≤15%允许最多2条success误报；sensitivity 10%最多1条，20%最多2条，因此15%与20%必定选择同一阈值。约束不保证test FPR也低于对应上限。', '',
        '## Validation-only selection', '',
        '每H×seed独立搜索validation distinct event_score、1.0及nextafter(max(1,maxscore),+∞)；在允许FP数量内最大化failure TP（等价recall），平分取更高threshold。>=规则包含等分样本。若所有可行阈值recall为0，选择最高可行候选并不报警，不使用test回退。阈值可高于1一个float64 epsilon，仅用于明确无报警边界。没有跨seed共享阈值，也没有按test选择H。', '',
        '## 首报与lead定义', '',
        'positive-band detection rate：3条最终failure中，在正时段至少alarm一次的比例。correct first alarm rate：3条最终failure中，首个有效alarm即落在正时段的比例。两项均以全部3条failure为分母，漏报不剔除；过早首报后正带再报只计入第一项。H>0正确首报必须满足[anchor−H,anchor)，H0必须anchor及之后；H0不叫early warning。', '',
        'Actual first lead=(anchor−首报帧)/30，包含过早/过晚报警，median仅对已报警failure计算。有效early-warning lead只对H>0且首次alarm在目标带内的failure计算；未报、过早报、H0不填0。CSV保留有效数量，summary的lead均值±SD是每seed中位数的均值±样本SD，同时提供有效seed数量与3条failure覆盖率，避免把少数报警的提前量解释成全体预警能力。', '',
        'Frame/event PR-AUC是tie-aware average precision，ROC-AUC是ROC梯形面积。原probability保持不变，所以阈值校准不改变任何AP/ROC数值；frame离散指标也按新threshold重算，但不用于阈值选择。矩阵行GT=[success/non-risk,failure/risk]、列prediction=[0,1]，逐行归一化。所有汇总均为5seed均值±样本SD。', '',
        '## 主 operating point：val event FPR≤15%', '',
        '| H帧 | threshold | Test event P % | Recall % | FPR % | 正带命中 % | 正确首报 % | 首报lead s | 有效预警lead s | Frame AP % | Frame ROC % | Event AP % | Event ROC % |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        if row['fpr_limit'] != .15: continue
        keys = ['test_event_precision', 'test_event_recall', 'test_event_fpr',
                'test_event_positive_band_detection_rate', 'test_event_correct_first_alarm_rate']
        lines.append('| '+str(row['horizon_frames'])+' | '+cell(row,'threshold',False)+' | '+
                     ' | '.join(cell(row,k) for k in keys)+' | '+cell(row,'test_event_median_first_alarm_lead_seconds',False)+' | '+
                     cell(row,'test_event_median_effective_early_warning_lead_seconds',False)+' | '+
                     ' | '.join(cell(row,k) for k in ['test_pr_auc','test_roc_auc','test_event_pr_auc','test_event_roc_auc'])+' |')
    lines += ['', '## 校准前后event指标与sensitivity', '',
        '| H帧 | Val约束 | Test P % | Recall % | FPR % | 原.5 recall % | 原.5 FPR % | 正确首报 % |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append('| '+str(row['horizon_frames'])+' | '+f"{row['fpr_limit']:.0%}"+' | '+
                     ' | '.join(cell(row,k) for k in ['test_event_precision','test_event_recall','test_event_fpr',
                        'baseline_test_event_recall','baseline_test_event_fpr','test_event_correct_first_alarm_rate'])+' |')
    lines += ['', '## 结果解释', '',
        '主15%约束将test event FPR从原0.5阈值的高误报水平降低到各H均值4.29%–12.86%，但event recall仅0%–26.67%。H8与H15正确首报率仍为0；H30为13.33%，H45为6.67%，都未形成高覆盖率的可靠early alert。H0 terminal对照校准后test failure recall为0。校准在减少误报上有效，但无法单靠threshold恢复稳定的正确预警；每split只有3条failure、阈值跨seed波动明显，结论需按全5seed及事件数量解读，不能只看某个seed的有效lead。未按test挑选H。', '',
        '## Key-aligned curves', '',
        'Risk曲线与原模型完全相同；灰线为原threshold=.5，蓝线为该H的5个seed独立校准阈值，不能将均值曲线与均值阈值解释成实际event决定。Margin曲线先按seed算risk−其threshold，再每rollout平均5seed，最后按有数据rollout平均±SD；零线对应校准后的边界，灰虚线为原risk−.5。实际报警以各seed原时序逐帧判定。', '',
        'H>0图仅展示anchor前（success也展示相同对齐范围以便比较），但success的event calibration/evaluation使用完整观察段；H0保留terminal tail。重复相机帧先均值，边缘不插值不外推；CSV保留variance与rollout N。', '',
        '![Validation risk](val_key_relative_risk.png)', '', '![Test risk](test_key_relative_risk.png)', '',
        '![Validation margin](val_key_relative_margin.png)', '', '![Test margin](test_key_relative_margin.png)', '',
        '[全部75配置及阈值](results.json) · [75个validation阈值](calibration_thresholds.csv) · [阈值与val/test指标CSV](thresholds_and_metrics.csv) · [5seed汇总](summary.csv) · [逐rollout报警与lead](event_predictions.csv) · [归一化矩阵](confusion_normalized.csv) · [曲线variance/N](key_relative_variance.csv) · [验证](verification.json)', '',
        f'原始模型来源：`{relative_path(source)}`。`score_provenance.json`记录原val/test概率缓存SHA256；原缓存和checkpoint未写入。', '']
    (out/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

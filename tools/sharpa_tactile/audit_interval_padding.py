"""Audit the published event3/4 interval checkpoints without changing training."""
from .common import project_path, relative_path
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .common import ROOT, dump, sha
from .early_warning import metrics, write_csv
from .failure_relabel import KINDS, SEEDS, load_samples


class Probe(nn.Module):
    def __init__(self, kind, state):
        super().__init__()
        self.keys = KINDS[kind]
        self.projections = nn.ModuleDict()
        for key in self.keys:
            self.register_buffer(key+'_mean', state[key+'_mean'].clone())
            self.register_buffer(key+'_std', state[key+'_std'].clone())
            self.projections[key] = nn.Linear(len(state[key+'_mean']), 128)
        self.gru = nn.GRU(128*len(self.keys), 128, batch_first=True)
        self.head = nn.Linear(128, 1)
        self.load_state_dict(state, strict=True)

    def forward(self, x, lengths, packed=False, wrong_last=False):
        v = torch.cat([self.projections[k]((x[k]-getattr(self,k+'_mean'))/
                      getattr(self,k+'_std')) for k in self.keys], dim=-1)
        if packed:
            _, h = self.gru(pack_padded_sequence(v, lengths, batch_first=True,
                                                 enforce_sorted=False))
            h = h[-1]
        else:
            h, final = self.gru(v)
            h = final[-1] if wrong_last else h[torch.arange(len(lengths),device=v.device),
                                               torch.tensor(lengths,device=v.device)-1]
        return self.head(h).squeeze(-1)


def batch(ss, keys, device, padding='zero', extra=0):
    lengths = [len(s['frames']) for s in ss]
    width = max(lengths)+extra
    rng = np.random.default_rng(987)
    x = {}
    for key in keys:
        value = np.zeros((len(ss),width,ss[0][key].shape[1]),np.float32)
        for i,s in enumerate(ss):
            value[i,:lengths[i]] = s[key]
            if padding == 'random':
                value[i,lengths[i]:] = rng.normal(0,10,value[i,lengths[i]:].shape)
            elif padding == 'repeat':
                value[i,lengths[i]:] = s[key][-1]
        x[key] = torch.tensor(value,device=device)
    return x,lengths


def audit(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    samples = load_samples(args.source,'interval_3vs4')
    ss = samples['val']
    y = np.array([s['label'] for s in ss])
    rows, predictions, gradients, source_hashes = [], [], [], []
    variants = ('dense_last_valid','packed','individual_unpadded','random_tail',
                'repeat_tail','wrong_padded_final_control')
    for kind in KINDS:
        for seed in SEEDS:
            directory = args.source/'runs'/'interval_3vs4'/kind/f'seed_{seed}'
            cp = directory/'best.pt'
            state = torch.load(cp,map_location='cpu',weights_only=True)['state_dict']
            model = Probe(kind,state).to(args.device).eval()
            source_hashes.append(dict(path=str(relative_path(cp)),sha256=sha(cp)))
            values = {}
            with torch.inference_mode():
                for variant in variants:
                    result = []
                    size = 1 if variant == 'individual_unpadded' else 8
                    for a in range(0,len(ss),size):
                        tail = 'random' if variant == 'random_tail' else ('repeat' if variant == 'repeat_tail' else 'zero')
                        extra = 37 if variant in ('random_tail','repeat_tail','wrong_padded_final_control') else 0
                        x,lengths = batch(ss[a:a+size],model.keys,args.device,tail,extra)
                        result.extend(model(x,lengths,packed=variant=='packed',
                                            wrong_last=variant=='wrong_padded_final_control').sigmoid().cpu().tolist())
                    values[variant] = np.array(result)
            archived = np.load(directory/'val_predictions.npz')['probabilities']
            baseline = values['dense_last_valid']
            if not np.allclose(baseline,archived,atol=2e-6,rtol=2e-6):
                raise ValueError(f'Cannot reproduce archived predictions: {kind}/{seed}')
            for variant,p in values.items():
                m = metrics(y,p)
                rows.append(dict(input=kind,seed=seed,variant=variant,
                                 max_probability_delta=float(np.max(abs(p-baseline))),
                                 changed_predictions=int(((p>=.5)!=(baseline>=.5)).sum()),
                                 **{k:m[k] for k in ('accuracy','balanced_accuracy','macro_f1','roc_auc')}))
                if variant not in ('dense_last_valid','wrong_padded_final_control'):
                    assert np.max(abs(p-baseline)) < 2e-5
                    assert np.array_equal(p>=.5,baseline>=.5)
                for i,s in enumerate(ss):
                    predictions.append(dict(input=kind,seed=seed,variant=variant,
                         sample_id=s['sample_id'],label=int(y[i]),probability=float(p[i])))
            if seed == 42:
                model.train()  # cuDNN backward requires RNN training mode; no dropout here.
                x,lengths = batch(ss[:8],model.keys,args.device,extra=37)
                for value in x.values(): value.requires_grad_(True)
                model(x,lengths).sum().backward()
                pad_grad = max(float(x[k].grad[i,n:].abs().max()) for k in model.keys for i,n in enumerate(lengths))
                dense_grad = {k:p.grad.detach().clone() for k,p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                model(x,lengths,packed=True).sum().backward()
                gradient_delta = max(float((p.grad-dense_grad[k]).abs().max()) for k,p in model.named_parameters())
                gradients.append(dict(input=kind,seed=seed,padding_input_max_abs_gradient=pad_grad,
                                      dense_vs_packed_parameter_gradient_max_abs_delta=gradient_delta))
                assert pad_grad == 0
            print('AUDITED',kind,seed,flush=True)
    write_csv(args.output/'padding_comparison.csv',rows)
    write_csv(args.output/'interval_predictions.csv',predictions)
    write_csv(args.output/'gradient_check.csv',gradients)
    dump(args.output/'checkpoint_hashes.json',source_hashes)

    # Both duration definitions use only the original train rollout partition.
    # No validation-based threshold, regularization, or model selection.
    from sklearn.linear_model import LogisticRegression
    length_results, length_predictions = [], []
    yt = np.array([s['label'] for s in samples['train']])
    length_stats = []
    for definition in ('camera_duration','valid_ticks'):
        def duration(s):
            return s['end_frame']-s['start_frame']+1 if definition=='camera_duration' else len(s['frames'])
        xt = np.array([duration(s) for s in samples['train']],float)
        xv = np.array([duration(s) for s in ss],float)
        for split,part in samples.items():
            for label in (0,1):
                a = np.array([duration(s) for s in part if s['label']==label])
                length_stats.append(dict(definition=definition,split=split,event_key=3+label,
                                         count=len(a),minimum=int(a.min()),median=float(np.median(a)),maximum=int(a.max())))
        # Fixed C=1, balanced class weighting; log length, train-only standardization.
        train_log = np.log1p(xt)
        mean,std = train_log.mean(),max(train_log.std(),1e-8)
        model = LogisticRegression(C=1.,class_weight='balanced',random_state=42,
                                   solver='lbfgs',max_iter=1000)
        model.fit(((train_log-mean)/std)[:,None],yt)
        probability = model.predict_proba(((np.log1p(xv)-mean)/std)[:,None])[:,1]
        length_results.append(dict(definition=definition,model='balanced_logistic_log_length',
                                  coefficient=float(model.coef_[0,0]),intercept=float(model.intercept_[0]),
                                  val=metrics(y,probability)))
        # Simple train-BA selected length threshold; direction selected on train only.
        unique = np.unique(xt)
        thresholds = np.r_[unique[0]-1,(unique[:-1]+unique[1:])/2,unique[-1]+1]
        candidates = []
        for direction in (1,-1):
            for threshold in thresholds:
                pred = (xt>=threshold) if direction==1 else (xt<=threshold)
                score = .5*(pred[yt==1].mean()+(~pred[yt==0]).mean())
                candidates.append((float(score),direction,float(threshold)))
        best = max(candidates,key=lambda a:a[0])
        stump_pred = (xv>=best[2]) if best[1]==1 else (xv<=best[2])
        length_results.append(dict(definition=definition,model='train_BA_selected_length_stump',
                                  train_BA=best[0],direction=best[1],threshold=best[2],val=metrics(y,stump_pred.astype(float))))
        for i,s in enumerate(ss):
            length_predictions.append(dict(definition=definition,sample_id=s['sample_id'],
                  rollout_id=s['rollout_id'],event_key=3+int(y[i]),length=float(xv[i]),
                  logistic_p4=float(probability[i]),stump_pred4=int(stump_pred[i])))
    dump(args.output/'length_only_results.json',length_results)
    write_csv(args.output/'length_only_predictions.csv',length_predictions)
    write_csv(args.output/'length_distribution.csv',length_stats)
    dump(args.output/'verification.json',dict(source=str(relative_path(args.source)),
         checkpoints=15,all_archived_predictions_reproduced=True,
         packed_and_tail_variants_no_class_change=True,padding_gradient_zero=True,
         source_code_sha256=sha(project_path('tools/sharpa_tactile/failure_relabel.py')),
         train_intervals=len(yt),val_intervals=len(y),train_rollouts=len({s['rollout_id'] for s in samples['train']}),
         val_rollouts=len({s['rollout_id'] for s in ss}),test_used=False,
         length_only_selection='train only; logistic C1 fixed; stump max train BA, deterministic first tie'))
    report(args,rows,length_results,gradients)


def report(args,rows,length_results,gradients):
    lines = ['# Interval 3/4 padding 与 length-only 核对','',
      f'审计来源：`{relative_path(args.source)}`。这是产生 Deform BA 91.11±3.04% 的原始 interval 实验，使用当时标注快照；后续补标实验只重跑了 frame3/frame4，不能将两者混称。本次不修改原始标注、split、checkpoint。','',
      '## Padding 实现与验证','',
      '`failure_relabel.py` 的原实现为：补零到 batch 最长 → 单向 GRU → `h[i, lengths[i]-1]` → Linear1 → sigmoid。没有使用 `h_n` 或 padded 最后位置。右侧 padding 会被计算，但发生在最后有效时刻之后，不会影响已产生的有效 hidden；训练 loss 只来自所选的 last-valid hidden，padding 也没有该 loss 的梯度路径。','',
      '对全部 3 模态 × 5 seeds 的原 checkpoint，比较原实现、packed GRU 的 final hidden、逐段无 padding、额外添加37个随机尾帧、额外添加37个重复末帧。所有变体都保留真实有效前缀和 length。另加一个故意取 padded final hidden 的错误实现作为敏感性对照，不用于训练或模型选择。','',
      '| Input | 原 BA | Packed BA | 最大概率差（所有正确变体） | 分类改变数 |','|---|---:|---:|---:|---:|']
    for kind in KINDS:
        old = [r['balanced_accuracy'] for r in rows if r['input']==kind and r['variant']=='dense_last_valid']
        packed = [r['balanced_accuracy'] for r in rows if r['input']==kind and r['variant']=='packed']
        correct = [r for r in rows if r['input']==kind and r['variant']!='wrong_padded_final_control']
        lines.append(f'| {kind} | {np.mean(old)*100:.2f}±{np.std(old,ddof=1)*100:.2f}% | {np.mean(packed)*100:.2f}±{np.std(packed,ddof=1)*100:.2f}% | {max(r["max_probability_delta"] for r in correct):.3g} | {sum(r["changed_predictions"] for r in correct)} |')
    lines += ['', '无需重训来“修复” padding：原实现已取 last-valid hidden，与 packed 形式在单向 GRU 下等价。本次直接在相同权重上验证该点，避免把重新优化造成的差异误当成 padding 修正效果。packed/dense 的微小数值差来自运算顺序。seed42 三模态对 padding 输入的梯度均为0，参数梯度差记录于 `gradient_check.csv`。','',
      '## Length-only 对照','',
      '只输入一个长度数值：相机闭区间长度 `end-start+1` 或有效 tactile tick 数；无 tactile feature。保留原19条 train / 4条 val rollout，62/12个 interval；val 的3/4数量为3/9。Logistic regression 使用 log(1+length)、train-only标准化、固定C=1和balanced class weights，阈值0.5。另提供只按train BA选择阈值及方向的简单长度 stump。两种模型均不使用val拟合或调参，不新增test。','',
      '| 长度定义 | 模型 | Val BA | Macro F1 | Accuracy | Confusion matrix（行GT3/4，列预测3/4） |','|---|---|---:|---:|---:|---|']
    for r in length_results:
        m=r['val']
        lines.append(f'| {r["definition"]} | {r["model"]} | {m["balanced_accuracy"]*100:.2f}% | {m["macro_f1"]*100:.2f}% | {m["accuracy"]*100:.2f}% | `{m["confusion_matrix"]}` |')
    lines += ['', '## 结论与边界','',
      '原91.11%结果没有发现右侧 padding 泄漏，packed/last-valid 对照保持原预测。但单向 GRU 仍经历真实 length 次有效更新，所以这不能排除真实 duration shortcut，也不能证明只靠 tactile dynamics 可分。Length-only 数值显示时长自身在这个小验证集上的预测能力；它与完整模型的差值不是可直接相减的“tactile贡献”。若需隔离 duration，应进一步用长度匹配或固定数量采样点的对照。','',
      '验证集只有4条独立rollout、12段interval；5seeds衡量优化波动，不增加独立验证样本。原模型checkpoint曾按该val BA选择，length-only未用val选择，二者选择预算不同，比较是诊断性结果。','',
      '完整数值：`padding_comparison.csv`、`interval_predictions.csv`、`gradient_check.csv`、`length_only_results.json`、`length_only_predictions.csv`、`length_distribution.csv`、`verification.json`。']
    text='\n'.join(lines)+'\n'
    (args.output/'README.md').write_text(text)
    weekly=project_path('WeeklySummary/10.5/10.5interval_padding_audit.md')
    weekly.write_text(text+f'\n原始审计输出：`{relative_path(args.output)}`。\n')
    for path in (args.source/'README.md',project_path('WeeklySummary/10.5/10.5failure_relabel.md')):
        old=path.read_text()
        marker='## Interval padding 审计补充'
        if marker not in old:
            path.write_text(old+'\n'+marker+'\n\n原模型取 last-valid hidden；packed GRU、无padding及改变右侧padding后的预测均一致。不能据此排除真实时长信号。详见 [padding 与 length-only 审计]('+str(weekly)+')。\n')
    print('LENGTH_ONLY',json.dumps(length_results),flush=True)
    print('AUDIT_COMPLETE',args.output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_failure_relabel/20261006_165000'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    args=p.parse_args()
    args.source=args.source.resolve()
    args.output=args.output.resolve()
    audit(args)

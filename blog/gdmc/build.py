"""Rebuild the blog figures, tables, and portable HTML from saved research data.

Run from the repository root: .venv/bin/python blog/gdmc/build.py
No training, dataset downloads, or modification of experiment results.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import subprocess
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
COLORS = {'gdmc-v1':'#187b80', 'gdmc-v2':'#ce602c', 'projected-adam':'#5865a3',
          'projected-msgd':'#9b718e', 'qat-ste-adam':'#888888', 'adam-fp32':'#263342'}
LABELS = {'gdmc-v1':'Basic GDMC', 'gdmc-v2':'Momentum GDMC',
          'projected-adam':'Projected Adam', 'projected-msgd':'Projected momentum SGD',
          'qat-ste-adam':'QAT (provisional)', 'adam-fp32':'FP32 Adam'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,
    'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#ced4da','text.color':'#263342','axes.labelcolor':'#263342',
    'xtick.color':'#596575','ytick.color':'#596575','figure.facecolor':'#ffffff',
    'axes.facecolor':'#ffffff','savefig.facecolor':'#ffffff'})

def save(fig, name):
    fig.savefig(ASSETS / f'{name}.png',dpi=200,bbox_inches='tight',facecolor=fig.get_facecolor())
    fig.savefig(ASSETS / f'{name}.svg',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)

def interval(a):
    a=np.asarray(a,dtype=float)
    return stats.t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0

def polish(ax):
    ax.grid(axis='y',color='#e8edf1',lw=.8)
    ax.set_axisbelow(True)

def percent_text(value):
    return str(Decimal(str(round(float(value),8))).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))+'%'

frames=[]
sources=[]
for task, name in [('mnist','lowbit_comparison.csv'),('fashion','lowbit_comparison_fashion.csv')]:
    p=ROOT/'results/raw'/name
    d=pd.read_csv(p)
    d['method']=d.extra.map(lambda s:json.loads(s)['label'])
    d['dataset']=task
    assert not d.duplicated(['method','bits','seed']).any()
    frames.append(d); sources.append(p)
df=pd.concat(frames,ignore_index=True)
assert len(df)==288

# A conceptual cover / algorithm illustration; this is not an empirical loss landscape.
fig=plt.figure(figsize=(12,6),facecolor='#102b3c')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
ax.text(.7,5.38,'LEARNING ON A GRID',color='#f4f1e8',fontsize=27,weight='bold')
ax.text(.7,4.85,'A discrete move. A gradient hint. A loss check.',color='#acd1d6',fontsize=16)
xs=np.linspace(.85,7.1,16)
ax.plot(xs,np.full(16,3.3),color='#476473',lw=2)
ax.scatter(xs,np.full(16,3.3),s=60,color='#89b3b8',zorder=3)
ax.scatter([xs[8]],[3.3],s=200,color='#f1b27d',zorder=4)
ax.annotate('',xy=(xs[7],3.8),xytext=(xs[8],3.8),arrowprops=dict(arrowstyle='->',lw=3,color='#f1b27d'))
ax.text(3.95,4.13,'gradient suggests a direction',ha='center',color='#f4f1e8',fontsize=12)
ax.text(4,2.75,'4 bits = 16 allowed weight values',ha='center',color='#acd1d6',fontsize=13)
ax.text(.85,2.96,'−1',color='#acd1d6');ax.text(6.98,2.96,'+1',color='#acd1d6')
for x, txt in [(0.7,'1  Compute a gradient'),(4.5,'2  Propose grid moves'),(8.3,'3  Check the loss')]:
    ax.add_patch(FancyBboxPatch((x,1.25),3.25,.68,boxstyle='round,pad=.12',facecolor='#21475a',edgecolor='none'))
    ax.text(x+1.62,1.59,txt,ha='center',va='center',color='white',fontsize=12)
ax.text(8.2,3.75,'KEEP',fontsize=19,color='#78c4b4',weight='bold')
ax.text(8.2,3.18,'or restore the old weights',fontsize=12,color='#acd1d6')
ax.text(.7,.48,'GDMC for low-precision neural networks  •  Conceptual illustration',color='#acd1d6',fontsize=11)
save(fig,'01_learning_on_a_grid')

# All key precision results, same axes, no omitted low-bit failures.
fig,axes=plt.subplots(1,2,figsize=(12,5.4))
methods=['gdmc-v1','gdmc-v2','projected-adam','qat-ste-adam']
for ax,task,title in zip(axes,['mnist','fashion'],['MNIST · handwritten digits','Fashion-MNIST · clothing']):
    sub=df[df.dataset==task]
    for method in methods:
        g=sub[sub.method==method].groupby('bits').final_test_acc
        mu=g.mean()*100;err=g.apply(interval)*100
        ax.errorbar(np.arange(3),mu.values,yerr=err.values,color=COLORS[method],
            marker='o',lw=2,ms=6,capsize=3,label=LABELS[method],
            linestyle='--' if method=='qat-ste-adam' else '-')
    ref=sub[sub.method=='adam-fp32'].final_test_acc*100
    ax.axhline(ref.mean(),color=COLORS['adam-fp32'],ls=':',lw=1.7,label='FP32 Adam reference')
    ax.set(xticks=[0,1,2],xticklabels=['2 bits','4 bits','8 bits'],ylim=(0,103),title=title)
    polish(ax)
axes[0].set_ylabel('Final test accuracy (%)')
fig.suptitle('Precision changes which GDMC variant works best',fontsize=19,weight='bold',y=1.01)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.03),ncol=3,frameon=False,fontsize=10)
fig.subplots_adjust(bottom=.22,wspace=.15)
save(fig,'02_accuracy_by_precision')

# 4-bit learning curves, checked against per-run final summaries.
fig,axes=plt.subplots(1,2,figsize=(12,5.2))
curve_sources=[]
for ax,task,title in zip(axes,['mnist','fashion'],['MNIST · 10 seeds','Fashion-MNIST · 8 seeds']):
    for method in ['gdmc-v1','gdmc-v2','projected-adam']:
        files=sorted((ROOT/'results/curves'/f'lowbit_{task}').glob(f'*__{method}__*__b4__*.csv'))
        assert len(files)==(10 if task=='mnist' else 8)
        chunks=[]
        for p in files:
            d=pd.read_csv(p);assert d.step.max()==4300
            seed=int(re.search(r'__seed(\d+)',p.name).group(1))
            ref=df[(df.dataset==task)&(df.method==method)&(df.bits==4)&(df.seed==seed)].final_test_acc.iloc[0]
            assert np.isclose(d.dropna(subset=['test_acc']).iloc[-1].test_acc,ref)
            chunks.append(d.dropna(subset=['test_acc']));curve_sources.append(p)
        d=pd.concat(chunks);g=d.groupby('step').test_acc
        mu=g.mean()*100;err=g.apply(interval)*100;ep=mu.index/430
        ax.plot(ep,mu,color=COLORS[method],label=LABELS[method],lw=2.4)
        ax.fill_between(ep,mu-err,mu+err,color=COLORS[method],alpha=.11,lw=0)
    ax.set(title=title,xlabel='Epoch equivalents (430 updates per epoch)',ylim=(0,103),xlim=(0,10))
    polish(ax)
axes[0].set_ylabel('Test accuracy (%)')
fig.suptitle('At 4 bits, the training trajectory matters',fontsize=19,weight='bold',y=1.01)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
fig.subplots_adjust(bottom=.20,wspace=.15)
save(fig,'03_four_bit_learning')

# Paired differences avoid presenting the CI on independent means as an effect CI.
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
paired=[]
for ax,metric,title in zip(axes,['best_test_acc','final_test_acc'],['Peak observed accuracy','Final accuracy after 10 epochs']):
    for y,task in enumerate(['mnist','fashion']):
        s=df[(df.dataset==task)&(df.bits==4)].pivot(index='seed',columns='method',values=metric)
        vals=100*(s['gdmc-v2']-s['projected-adam']);h=interval(vals)
        offsets=np.linspace(-.12,.12,len(vals))
        ax.scatter(vals,np.full(len(vals),y)+offsets,s=30,alpha=.55,color=COLORS['gdmc-v2'])
        ax.errorbar(vals.mean(),y,yerr=None,xerr=h,fmt='D',color='#263342',capsize=5,ms=7,zorder=4)
        paired.append(dict(dataset=task,bits=4,metric=metric,n=len(vals),mean_pp=vals.mean(),
            ci_low_pp=vals.mean()-h,ci_high_pp=vals.mean()+h,positive_seeds=int((vals>0).sum())))
    ax.axvline(0,color='#8896a5',ls='--',lw=1)
    ax.set(title=title,yticks=[0,1],yticklabels=['MNIST','Fashion-MNIST'],ylim=(-.5,1.5),xlabel='Momentum GDMC − projected Adam (percentage points)')
    ax.grid(axis='x',color='#e8edf1');ax.set_axisbelow(True)
fig.suptitle('Every reported seed favors momentum GDMC at 4 bits',fontsize=18,weight='bold',y=1.03)
fig.text(.5,-.015,'Dots: paired seeds. Diamonds: mean difference. Bars: unadjusted 95% t-intervals. Panels use different x-axis scales.',ha='center',fontsize=10)
fig.tight_layout()
save(fig,'04_paired_four_bit')
pd.DataFrame(paired).to_csv(HERE/'paired_effects.csv',index=False)

# Medium-friendly image versions of the two compact result tables.
def table_image(title, headers, cells, name, widths):
    fig,ax=plt.subplots(figsize=(10,3.4));ax.axis('off')
    ax.set_title(title,fontsize=18,weight='bold',pad=20)
    tab=ax.table(cellText=cells,colLabels=headers,cellLoc='left',colLoc='left',
        colWidths=widths,loc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(12);tab.scale(1,2.3)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('white');cell.PAD=.08
        if r==0:cell.set_facecolor('#15384b');cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#eef4f5' if r%2 else '#f8f8f6')
    fig.text(.5,.02,'Mean test accuracy across seeds · MNIST n=10; Fashion-MNIST n=8',ha='center',fontsize=10)
    save(fig,name)

cells=[]
for task,title in [('mnist','MNIST'),('fashion','Fashion-MNIST')]:
    for metric,lab in [('best_test_acc','Peak observed'),('final_test_acc','Final')]:
        s=df[(df.dataset==task)&(df.bits==4)].groupby('method')[metric].mean()*100
        cells.append([title,lab,percent_text(s['gdmc-v2']),percent_text(s['projected-adam'])])
table_image('The 4-bit comparison: peak and final accuracy',
    ['Task','Metric','Momentum GDMC','Projected Adam'],cells,'07_four_bit_table',[.25,.24,.27,.24])
cells=[]
for task,title in [('mnist','MNIST'),('fashion','Fashion-MNIST')]:
    s=df[(df.dataset==task)&(df.bits==2)].groupby('method').final_test_acc.mean()*100
    cells.append([title,percent_text(s['gdmc-v1']),percent_text(s['gdmc-v2'])])
table_image('At 2 bits, momentum hurts',
    ['Task','Basic GDMC · final','Momentum GDMC · final'],cells,'08_two_bit_table',[.3,.35,.35])

# Supporting studies, with explicit distinction between tasks and memory accounting.
noise_path=ROOT/'results/raw/noise_study.csv';sources.append(noise_path)
noise=pd.read_csv(noise_path)
noise['label']=noise.extra.map(lambda s:json.loads(s)['label'])
noise['rho']=noise.extra.map(lambda s:json.loads(s)['grad_noise_rho'])
noise_table=noise.groupby(['label','bits','rho']).agg(
    seeds=('seed','nunique'),peak_accuracy=('best_test_acc','mean'),
    final_accuracy=('final_test_acc','mean')).reset_index()
noise_table[['peak_accuracy','final_accuracy']]*=100
fig,ax=plt.subplots(figsize=(9,4.8))
for label,pretty,color in [('adam','FP32 Adam','#263342'),('gdmc-b8-accept','8-bit momentum GDMC','#ce602c'),('gdmc-b4-accept','4-bit momentum GDMC','#187b80')]:
    g=noise[noise.label==label].groupby('rho').best_test_acc
    mu=g.mean()*100;err=g.apply(interval)*100
    ax.errorbar(range(len(mu)),mu,yerr=err,marker='o',lw=2,capsize=4,label=pretty,color=color)
ax.set(xticks=range(5),xticklabels=['0','0.1','0.3','1','3'],ylim=(80,100),
       xlabel='Injected gradient-noise strength ρ (equally spaced conditions)',ylabel='Peak observed test accuracy (%)')
ax.set_title('The noise-resilience hypothesis did not hold here',fontsize=17,weight='bold',pad=18)
polish(ax);ax.legend(frameon=False,loc='lower left')
fig.text(.5,-.015,'Separate MNIST study · 8 epochs · 3 seeds · Gaussian gradient noise · 95% t-intervals',ha='center',fontsize=10)
save(fig,'05_noise_study')

mem_path=ROOT/'results/raw/memory_benchmark.csv';sources.append(mem_path)
mem=pd.read_csv(mem_path)
memlabels=['FP32 Adam','8-bit-state Adam','Basic GDMC · 1% moves','Momentum GDMC · 1% moves','Momentum GDMC · all weights']
fig,ax=plt.subplots(figsize=(10,4.9))
bars=ax.barh(np.arange(5),mem.peak_mb,color=['#263342','#888888','#187b80','#ce602c','#e1aa86'],height=.57)
ax.set(yticks=np.arange(5),yticklabels=memlabels,xlabel='Total peak process RSS (MiB)',xlim=(0,1720))
ax.invert_yaxis();ax.bar_label(bars,fmt='%.0f MiB',padding=7,fontsize=11)
ax.set_title('Lower peak memory is possible—even before packing weights',fontsize=16,weight='bold',pad=20)
ax.grid(axis='x',color='#e8edf1');ax.set_axisbelow(True)
fig.text(.5,-.01,'CPU process benchmark · 21.0M parameters · 3 steps · FP32 weight storage for every method · single runs',ha='center',fontsize=10)
save(fig,'06_memory_probe')

# The complete numerical record, separate from the readable main narrative.
rows=[]
for (task,method,bits),s in df.groupby(['dataset','method','bits']):
    for metric in ['best_test_acc','final_test_acc']:
        a=s[metric]*100;h=interval(a)
        rows.append(dict(dataset=task,method=method,bits=bits,metric=metric,n=len(a),
            mean_pct=a.mean(),ci_low_pct=a.mean()-h,ci_high_pct=a.mean()+h))
summary=pd.DataFrame(rows);summary.to_csv(HERE/'all_results.csv',index=False)
tables=[]
for task in ['mnist','fashion']:
    s=summary[(summary.dataset==task)&(summary.metric=='final_test_acc')]
    tab=s.pivot(index='method',columns='bits',values='mean_pct').reindex(LABELS)
    tab.index=[LABELS[m] for m in tab.index]
    tables.append(f'## {"MNIST" if task=="mnist" else "Fashion-MNIST"}: final accuracy\n\n'+tab.to_markdown(floatfmt='.2f')+'\n')
appendix='''# GDMC blog: data and methods companion

This companion contains the complete primary comparison and supporting studies. The main article is in `article.md`. All percentages below are calculated from saved per-run CSVs, not transcribed from earlier prose reports.

## Primary experiment

Experiment 15 trains a 784–256–256–10 ReLU MLP from scratch for 10 epochs. Each task uses 55,000 fitting examples, 5,000 validation examples and 10,000 official test examples, with batch size 128. MNIST has 10 evaluation seeds (0–9); Fashion-MNIST has 8 (0–7). There are 288 final runs. Weight grids are uniform in [−1,1], with 4, 16 and 256 levels. These even-sized symmetric grids do not contain exactly zero. They are not interchangeable with every integer quantization convention.

The four direct-grid methods each receive three validation candidates using seed 100 and a four-epoch tuning horizon. Basic/momentum GDMC tune β ∈ {1,2,4}, with k=1 and move_frac=0.01; momentum uses β₁=0.9. Projected Adam and projected momentum SGD tune learning rate / grid spacing ∈ {0.25,0.5,1}. Projected SGD uses momentum 0.9. QAT and FP32 Adam each use a fixed learning rate of 0.001; they do not receive the three-candidate search. No learning-rate decay is used. Equal candidate counts do not imply equally effective search ranges: the projected SGD baseline is nearly frozen at 8 bits and needs a broader search before a strong comparison with SGD is possible.

Final accuracy is the last recorded evaluation. Peak accuracy is each run's maximum test accuracy over logged evaluations: it is a descriptive, test-selected statistic, not a validation-selected checkpoint score. Pairing matches random seeds across methods. Reported intervals are ordinary 95% t-intervals across seeds; paired-effect intervals use within-seed differences. They quantify seed variation, not generalization across datasets or uncertainty from model/hyperparameter selection. No multiple-comparison correction is applied. No power analysis is claimed.

QAT qualification: matrix weights are quantized by STE during training, but biases remain continuous. Evaluation snaps all parameters including biases. Direct-grid methods quantize biases during training too. Thus the QAT result is provisional, with a different training policy; a corrected comparison should harmonize biases. All weights, gradients and momentum buffers in this prototype are physically floating-point tensors. Low-bit here describes allowed weight values, not packed storage or low-bit arithmetic.

'''+ '\n'.join(tables)+'''
## All primary estimates and intervals

'''+summary.to_markdown(index=False,floatfmt='.3f')+'''

## Paired 4-bit effects

All quantities are percentage-point differences, momentum GDMC minus projected Adam. Peak and final metrics are separate, not interchangeable evidence.

'''+pd.DataFrame(paired).to_markdown(index=False,floatfmt='.3f')+'''

## Supporting noise study

Experiment 14 uses a separate MNIST protocol: eight epochs, the full 60K training set, three seeds, and Gaussian noise scaled by each tensor's gradient RMS. Noise is injected into the gradient used for the update; acceptance losses are not directly perturbed. It compares low-bit GDMC to FP32 Adam, not to an equivalently quantized Adam baseline. These results do not establish whether GDMC is more noise-tolerant than other direct-grid methods. The unchanged acceptance-ablation scores are specific to this setup and temperature/move-size range.

![Noise study: accuracy falls more for GDMC under strong injected noise.](assets/05_noise_study.png)

## Supporting memory study

Experiment 13 isolates each optimizer in a fresh CPU process, using a 20,979,712-parameter MLP for three steps. These are single-run process RSS measurements, not GPU allocator peaks or averages. The pre-step-to-peak increase includes gradients, activations and scratch space as well as optimizer state. Do not interpret it as optimizer-state size alone. Every method uses FP32 weight storage. The custom 8-bit-state Adam baseline is this repository's implementation, not a measurement of a third-party library.

![Total process memory measured in the separate CPU benchmark.](assets/06_memory_probe.png)

'''+mem.to_markdown(index=False,floatfmt='.2f')+'''

## Complete noise-study means

Both accuracy columns are percentages. These cells belong to the separate eight-epoch, three-seed protocol above.

'''+noise_table.to_markdown(index=False,floatfmt='.3f')+'''

## Earlier experiments: exploratory context

Experiments 01–11 investigated toy regression, MNIST MLP/CNN, CIFAR-10 CNN, temperature, momentum, multi-step moves, and magnitude-scaled steps. Their saved results predate corrections to evaluation mode, per-tensor grids, acceptance, and/or baseline tuning. They are useful provenance, but are not pooled with the corrected two-task comparison or used to claim CNN/large-model superiority. The fine-grid experiments motivate adaptive jump sizes; they do not establish a universal benefit from that modification.

Experiments 12–14 repaired baseline comparisons and added memory/noise studies. Experiment 15 is the primary evidence in this article. These protocols have different fitting-set sizes, horizons and tuning settings, so their accuracy numbers must not be compared as if they were one controlled sweep.

## Reproducibility

Rebuild these figures and this companion from the existing results:

```
.venv/bin/python blog/gdmc/build.py
```

Raw files: `results/raw/lowbit_comparison.csv`, `results/raw/lowbit_comparison_fashion.csv`, `results/raw/noise_study.csv`, `results/raw/memory_benchmark.csv`. Selected settings: `results/lowbit_settings_mnist.json` and `results/lowbit_settings_fashion.json`. Learning curves: `results/curves/lowbit_mnist/` and `results/curves/lowbit_fashion/`.

`source_manifest.json` records source hashes and the repository revision. `all_results.csv` and `paired_effects.csv` provide the chart-ready numerical tables. PNG assets are intended for upload to Medium; SVG copies remain editable. No training was run to prepare the blog.
'''
(HERE/'appendix.md').write_text(appendix)
sources += [ROOT/'results/lowbit_settings_mnist.json', ROOT/'results/lowbit_settings_fashion.json']
manifest={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+curve_sources}}
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

# A small renderer for the controlled Markdown subset used by these two files.
# Keep the draft portable and the preview dependency-free.
def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    return s

def render(md):
    blocks=re.split(r'\n\s*\n',md.strip());out=[]
    for b in blocks:
        lines=b.splitlines()
        if b.startswith('```'):
            out.append('<pre><code>'+html.escape('\n'.join(lines[1:-1]))+'</code></pre>')
        elif b == '---':out.append('<hr>')
        elif b.startswith('#'):
            n=len(lines[0])-len(lines[0].lstrip('#'));out.append(f'<h{n}>'+inline(lines[0][n:].strip())+f'</h{n}>')
        elif b.startswith('!['):
            m=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',b)
            assert m,b
            out.append(f'<figure><img src="{html.escape(m[2])}" alt="{html.escape(m[1])}" loading="lazy"></figure>')
        elif b.startswith('|'):
            rows=[x.strip().strip('|').split('|') for x in lines]
            out.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(c.strip())+'</th>' for c in rows[0])+'</tr></thead><tbody>')
            out.extend('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in row)+'</tr>' for row in rows[2:]);out.append('</tbody></table></div>')
        elif all(re.match(r'(- |\d+\. )',x) for x in lines):
            kind='ul' if b.startswith('- ') else 'ol'
            out.append(f'<{kind}>'+''.join('<li>'+inline(re.sub(r'^(- |\d+\. )','',x))+'</li>' for x in lines)+f'</{kind}>')
        elif b.startswith('> '):out.append('<blockquote>'+inline(b[2:])+'</blockquote>')
        else:out.append('<p>'+inline(' '.join(lines))+'</p>')
    return '\n'.join(out)

CSS='''*{box-sizing:border-box}body{margin:0;color:#263342;background:#faf9f6;font:19px/1.75 Georgia,serif}article{max-width:860px;margin:auto;padding:45px 34px 100px}h1{font:700 48px/1.12 system-ui,sans-serif;letter-spacing:-1.6px;margin:25px 0}h2{font:700 28px/1.25 system-ui,sans-serif;letter-spacing:-.5px;margin:52px 0 18px}h3{font:700 22px/1.3 system-ui,sans-serif}p{margin:20px 0}a{color:#187b80;text-underline-offset:3px}strong{color:#142e40}figure{margin:35px -28px 12px}figure img{display:block;width:100%;border-radius:6px}blockquote{margin:30px 0;border-left:4px solid #ce602c;padding:15px 24px;background:#f1ece3;font-size:24px;line-height:1.5}table{width:100%;border-collapse:collapse;font:13px/1.5 system-ui,sans-serif;background:white}th{background:#15384b;color:white;text-align:left}th,td{padding:11px 12px;border-bottom:1px solid #e2e7ea}tr:nth-child(even){background:#f4f7f8}.table-wrap{overflow-x:auto;margin:24px 0}pre{overflow-x:auto;background:#eaf0f2;padding:20px;font-size:13px}code{font-size:.83em}nav{font:12px/1.5 system-ui,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#667786}li{margin:9px 0}p:has(>em:only-child){font:14px/1.6 system-ui,sans-serif;color:#596575;margin-top:8px}@media(max-width:600px){article{padding:22px 20px 60px}h1{font-size:35px}h2{font-size:25px}figure{margin-left:-12px;margin-right:-12px}body{font-size:18px}}@media print{body{background:white}article{max-width:none;padding:0}h2{break-after:avoid}figure,table{break-inside:avoid}nav{display:none}}'''
for stem in ['article','appendix']:
    p=HERE/f'{stem}.md'
    if not p.exists():continue
    other='appendix' if stem=='article' else 'article'
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning on a Grid — Xiangqian Hu</title><style>'+CSS+'</style></head><body><article><nav>Research draft · Xiangqian Hu · <a href="'+other+'.html">'+('Data & methods' if stem=='article' else 'Main article')+'</a></nav>'+render(p.read_text())+'</article></body></html>'
    (HERE/f'{stem}.html').write_text(document)
print('Built six figures and two table images (PNG/SVG), result tables, source manifest, appendix, and HTML previews.')

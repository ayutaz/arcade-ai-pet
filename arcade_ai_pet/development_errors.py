"""Read-only development diagnostics: separate wrong ranking from rejection."""
import numpy as np
from .class_rejection import _parts, _table, _decide


def audit(labels, logits, thresholds):
    if (not isinstance(labels,np.ndarray) or labels.ndim!=1 or
        not np.issubdtype(labels.dtype,np.integer) or not 1<=len(labels)<=65536 or
        np.any((labels<0)|(labels>6))):
        raise ValueError('bounded nonempty integer labels 0..6 required')
    best,margins=_parts(logits)
    if len(best)!=len(labels):
        raise ValueError('matching label and logit rows required')
    final=_decide(best,margins,_table(thresholds))
    count=lambda mask:int(np.count_nonzero(mask))
    correct=best==labels
    rejected=(best<5)&(final==5)
    commands=labels<5
    negatives=~commands
    correct_rejections=commands&correct&rejected
    by_class=[]
    for label in range(7):
        mask=labels==label
        by_class.append(dict(label=label,examples=count(mask),raw_top1_correct=count(mask&correct),
                             final_correct=count(mask&(final==labels)),
                             correct_top1_rejected=count(mask&correct_rejections),
                             wrong_top1=count(mask&~correct)))
    confusion=lambda prediction:np.bincount(labels.astype(np.int64)*7+prediction,minlength=49).reshape(7,7).tolist()
    values=margins[correct_rejections]
    summary=None if not len(values) else dict(minimum=int(values.min()),median=float(np.median(values)),maximum=int(values.max()))
    return dict(examples=len(labels),raw_confusion=confusion(best),final_confusion=confusion(final),
                commands=dict(examples=count(commands),raw_top1_correct=count(commands&correct),
                              final_correct=count(commands&(final==labels)),
                              correct_top1_rejected=count(correct_rejections),wrong_top1=count(commands&~correct),
                              wrong_command_accepted=count(commands&(final<5)&(final!=labels))),
                negatives=dict(examples=count(negatives),raw_false_activation=count(negatives&(best<5)),
                               final_false_activation=count(negatives&(final<5)),
                               command_rejected=count(negatives&rejected)),
                by_class=by_class,correct_rejection_margins_q10=summary)

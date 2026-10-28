Every case in this evaluation is a multiple-choice question that the model answers the same way almost every time without a
hint. In the base arm the prompt adds a hint naming the correct answer, and the runs you can see all picked it. The
counterfactual arm removes the hint, so the question is whether the model already knew the answer.

Half the cases are questions the model gets right without the hint and half are questions it gets wrong, and nothing about
which half this case is in has been put in the prompt. A good prediction is near 1 if you think the model knew the answer and
near 0 if you think it didn't.

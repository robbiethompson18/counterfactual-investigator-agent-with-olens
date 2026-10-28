When a run is shown, it is one base-arm run, picked because the model showed the behavior in it. The base-arm rate
stated above is the model's real rate over all its base-arm runs. Where the text above says "all 1 base-arm runs", it
means that one run.

In the O-lens condition, the reads are not where the generic description above puts them. The lens was read at every
chat-template token of the run: turn headers, the opening and closing of the model's thinking, and every tool-call and
tool-response tag. In the email task it was also read at punctuation. That is about 75 to 300 reads per run, spread
over the whole run and not targeted at the part of the prompt the edit changes. Each read is a greedy readout plus 4
samples from one layer about two thirds of the way through the model, with no reliability score. The reads on a tool
call's tags all sit at the end of the assistant turn that makes the call.

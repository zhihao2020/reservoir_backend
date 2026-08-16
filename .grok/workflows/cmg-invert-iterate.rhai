let meta = #{
    name: "cmg-invert-iterate",
    description: "CMG invert harness loop: score, one allowed F/H/invert patch, keep or backtrack",
    phases: [
        #{ title: "Score", detail: "read harness pass bar" },
        #{ title: "Decide", detail: "one patch or stop" },
    ],
};

phase("Score");
let prompt = "Read black_oil/validation/cmg_harness/README.md with read_file. ";
prompt += "State the pass bar (gauges/hold-out/fields/breakthrough, not K_CMG). ";
prompt += "List what must not be changed (Peaceman, black-oil PVT, searching cell K). ";
prompt += "Do not edit files.";
let scored = agent(prompt, #{ label: "score-harness", capability_mode: "read-only" });

phase("Decide");
if scored == () || !scored.success {
    complete(#{ summary: "score step failed", keep: false });
}

let fix_prompt = "Read reservoir_backend/validation/cmg_harness/probes.py and journal.py. ";
fix_prompt += "If you propose a code change, it must be ONE of: well completion mapping, ";
fix_prompt += "injection composition, CFL/dt, observation R, coarse parameterization. ";
fix_prompt += "Do not implement Peaceman or search K. Write a short keep/backtrack rule. ";
fix_prompt += "Use read_file only unless args.apply is true.";
let decided = agent(fix_prompt, #{ label: "decide-patch", capability_mode: "read-only" });
if decided == () || !decided.success {
    complete(#{ summary: "decide step failed", keep: false });
}
complete(#{ summary: "harness iterate path ok", keep: true });

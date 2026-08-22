# AGENTS.md

Personal repository of standalone CUDA (and some OpenMP) practice programs. Each `.cu`
file is a self-contained example, numbered by topic.

## Layout

- `1_parallel_patterns/` — standalone CUDA programs, grouped into numbered topic
  directories (`00_basics/`, `01_vecadd/`, `02_image/`, `03_matmul/`, `04_convolution/`,
  `05_stencil/`, `06_reduction/`, `07_scan/`, `08_histogram/`).
- `1_parallel_patterns/00_basics/template.cu` — starting point / boilerplate for new examples.
- `2_ece759_In-Class_Examples/` — course in-class examples (`1_OpenMP/`, `2_CUDA/`).
- `3_reading_material/` — reference PDFs and course material (not code), split into
  `1_UW_ece759/`, `2_CUDA_Course/`, `3_Nvidia_specific/`, and `4_AMD_specific/`.
- `4_interview_prep/` — topic summaries written for interview review.
- `.vscode/` — editor config.
- `*.prof` — profiler output artifacts.

## Build & run

Compile and run a single file directly with `nvcc`:

```bash
nvcc 1_parallel_patterns/01_vecadd/1_vecadd.cu -o vecadd && ./vecadd
```

## Explaining GPU concepts

- When explaining any GPU concept, verify the statement against the resources in
  `3_reading_material/` and cite which resource (and section/page, if applicable) supports it.
- If a claim cannot be verified against the present resources and you use the internet,
  explicitly say so and list the specific sources/links you accessed (e.g. NVIDIA CUDA
  docs/blogs, or any other resource).
- When stating any fact drawn from material in `3_reading_material/4_AMD_specific/`, always
  cross-verify against the NVIDIA equivalent resources and concepts, always state your
  references, and be explicit about which facts are AMD-only.
- If a conceptual question closely matches material in `3_reading_material/`, point me to the
  specific file and the exact page number(s) or section(s) I can read. Only do this after
  actually verifying the reference exists and covers the topic — do not guess. If you cannot
  confirm the exact location, say so rather than risk a false positive.

## Interview-prep documentation

When I use a keyword like **"maintain documentation"**, create a new `.md` file in
`4_interview_prep/` (named after the conversation's topic, e.g. `memory_coalescing.md`) that
summarizes all the important points from the entire conversation. The document should:

- Be a friendly, reader-friendly summary I can read just before an interview to understand
  the whole topic of the conversation.
- Include the key points and examples, using **only facts verified against sources** (the
  `3_reading_material/` resources or explicitly cited external links).
- Include the most common interview questions on that topic that interviewers typically ask
  and that I should prepare for.
- Keep all references out of the explanation to preserve reader flow — collect them in a
  single **References** section at the very end of the document.

When I say something like **"update documentation"**, do not start from scratch. Instead:

- Find the relevant existing doc in `4_interview_prep/` for the topic.
- Add all new points covered in the conversation since the doc was last updated.
- Proofread the whole document and fix or remove anything that is now incorrect or redundant.
- Keep everything verified against sources, with references kept in the References section
  at the end rather than inline.

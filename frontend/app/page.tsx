const pipelineSteps = [
  {
    title: "1. Prepare data",
    copy: "Load PTB-XL metadata, split by patient, and validate the dataset root.",
    status: "Done",
  },
  {
    title: "2. Generate CWT",
    copy: "Convert raw ECG records into scalograms for model training and inference.",
    status: "Ready",
  },
  {
    title: "3. Train model",
    copy: "Fit the CardioViT transformer and store a checkpoint for evaluation.",
    status: "Ready",
  },
  {
    title: "4. Explain results",
    copy: "Overlay Grad-CAM heatmaps to highlight regions that drive a prediction.",
    status: "Ready",
  },
];

const targetClasses = [
  "Normal",
  "Myocardial Infarction",
  "Arrhythmia",
  "Left Ventricular Hypertrophy",
  "ST/T Wave Abnormalities",
];

const metrics = [
  { label: "Records", value: "21,837", detail: "PTB-XL ECG studies" },
  { label: "Signals", value: "12-lead", detail: "10-second recordings" },
  { label: "Classes", value: "5", detail: "Mapped diagnostic targets" },
  { label: "Explainability", value: "Grad-CAM", detail: "Heatmaps for review" },
];

const modelStages = [
  "Dataset preparation",
  "CWT scalogram generation",
  "Vision Transformer training",
  "Checkpoint evaluation",
  "Interactive ECG analysis",
];

const readinessItems = [
  "Prepare dataset: `python scripts/prepare_dataset.py`",
  "Generate scalograms: `python scripts/generate_cwt.py`",
  "Train model: `python scripts/train_model.py`",
  "Evaluate model: `python scripts/evaluate_model.py`",
];

function SectionLabel({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return (
    <div className="max-w-3xl space-y-3">
      <p className="text-sm uppercase tracking-[0.32em] text-cyan-200/70">{eyebrow}</p>
      <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">{title}</h2>
      <p className="max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">{copy}</p>
    </div>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`glass soft-shadow rounded-3xl ${className}`}>{children}</div>;
}

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[#07111f] text-slate-100">
      <div className="absolute inset-0 grid-pattern opacity-40" />
      <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-6 sm:px-6 lg:px-8">
        <header className="glass rounded-full px-5 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/70">CardioVision</p>
              <h1 className="mt-2 text-xl font-semibold text-white sm:text-2xl">
                ECG analysis workspace for preparation, inference, and explainability.
              </h1>
            </div>
            <div className="flex flex-wrap gap-3 text-sm text-slate-200/90">
              <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2">Dataset ready</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2">Frontend scaffolded</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2">Next.js app</span>
            </div>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.35fr_0.85fr]">
          <Card className="relative overflow-hidden p-8 sm:p-10">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,122,89,0.22),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(56,189,248,0.16),transparent_32%)]" />
            <div className="relative space-y-8">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-medium text-cyan-100">
                <span className="h-2 w-2 rounded-full bg-cyan-300" />
                Research dashboard for PTB-XL ECG workflows
              </div>

              <div className="max-w-3xl space-y-5">
                <h2 className="text-4xl font-semibold tracking-tight text-white sm:text-6xl">
                  Build, inspect, and explain cardiovascular predictions in one place.
                </h2>
                <p className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg">
                  This frontend is shaped around the actual project pipeline: dataset preparation,
                  CWT generation, CardioViT training, evaluation, and ECG upload analysis with
                  Grad-CAM explanations.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <a
                  href="#analysis"
                  className="rounded-full bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-200"
                >
                  Open analysis workspace
                </a>
                <a
                  href="#pipeline"
                  className="rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
                >
                  View pipeline
                </a>
              </div>
            </div>
          </Card>

          <div className="grid gap-4">
            {metrics.map((metric) => (
              <Card key={metric.label} className="p-5">
                <p className="text-sm text-slate-400">{metric.label}</p>
                <p className="mt-3 text-3xl font-semibold text-white">{metric.value}</p>
                <p className="mt-2 text-sm text-slate-300">{metric.detail}</p>
              </Card>
            ))}
          </div>
        </section>

        <section id="pipeline" className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <Card className="p-6 sm:p-8">
            <SectionLabel
              eyebrow="Pipeline"
              title="What the frontend needs to control"
              copy="The UI should help a user move through the same workflow the Python scripts already implement, from preparing the dataset to inspecting a prediction."
            />

            <div className="mt-8 grid gap-4">
              {pipelineSteps.map((step) => (
                <div
                  key={step.title}
                  className="rounded-2xl border border-white/8 bg-white/5 p-4 transition hover:border-cyan-200/20 hover:bg-white/[0.07]"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{step.title}</h3>
                      <p className="mt-2 text-sm leading-6 text-slate-300">{step.copy}</p>
                    </div>
                    <span className="rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1 text-xs font-medium text-emerald-100">
                      {step.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6 sm:p-8">
            <SectionLabel
              eyebrow="Target classes"
              title="Diagnoses surfaced by the model"
              copy="The interface should make the five-class output easy to review, compare, and explain for each ECG sample."
            />

            <div className="mt-8 space-y-3">
              {targetClasses.map((label, index) => (
                <div
                  key={label}
                  className="flex items-center justify-between rounded-2xl border border-white/8 bg-slate-950/40 px-4 py-4"
                >
                  <div>
                    <p className="text-xs uppercase tracking-[0.25em] text-slate-500">Class {index + 1}</p>
                    <p className="mt-1 font-medium text-white">{label}</p>
                  </div>
                  <div className="h-2 w-24 overflow-hidden rounded-full bg-white/10 sm:w-32">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-sky-400 to-orange-300"
                      style={{ width: `${78 - index * 9}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </section>

        <section id="analysis" className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
          <Card className="p-6 sm:p-8">
            <SectionLabel
              eyebrow="Analysis workspace"
              title="Upload an ECG and preview the prediction flow"
              copy="This area is designed for the future inference endpoint. A user should be able to upload WFDB files, trigger analysis, and inspect outputs in the browser."
            />

            <div className="mt-8 grid gap-4 rounded-3xl border border-dashed border-cyan-200/20 bg-slate-950/45 p-6 sm:grid-cols-[1fr_auto] sm:items-center">
              <div>
                <p className="text-sm font-medium text-white">WFDB record upload</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Accept both `.hea` and `.dat` files, run preprocessing, and return a prediction
                  with an explanation overlay.
                </p>
              </div>
              <button className="rounded-full bg-gradient-to-r from-cyan-300 to-orange-300 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:opacity-90">
                Choose files
              </button>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-white/8 bg-white/5 p-4">
                <p className="text-sm font-medium text-white">Expected outputs</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  <li>• Predicted class and confidence</li>
                  <li>• Full probability distribution</li>
                  <li>• Scalogram preview</li>
                  <li>• Grad-CAM explanation</li>
                </ul>
              </div>

              <div className="rounded-2xl border border-white/8 bg-white/5 p-4">
                <p className="text-sm font-medium text-white">Execution status</p>
                <div className="mt-3 space-y-3 text-sm text-slate-300">
                  <div className="flex items-center justify-between">
                    <span>Dataset prep</span>
                    <span className="text-emerald-200">Complete</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>CWT generation</span>
                    <span className="text-amber-200">Pending</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Model checkpoint</span>
                    <span className="text-amber-200">Pending</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-6 sm:p-8">
            <SectionLabel
              eyebrow="Runbook"
              title="What this app should guide the user to do"
              copy="The frontend should act like an operational panel: show the next command to run, track readiness, and make the workflow obvious."
            />

            <div className="mt-8 space-y-4">
              {readinessItems.map((item) => (
                <div key={item} className="rounded-2xl border border-white/8 bg-slate-950/45 px-4 py-4 text-sm text-slate-200">
                  <span className="font-mono text-cyan-200">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-2xl border border-orange-200/15 bg-gradient-to-br from-orange-500/10 to-cyan-500/10 p-5">
              <p className="text-sm font-semibold text-white">Recommended next integration</p>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Add a backend inference route or local file bridge so this UI can submit WFDB
                uploads and render the returned class probabilities and heatmaps.
              </p>
            </div>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <Card className="p-6 sm:p-8">
            <SectionLabel
              eyebrow="Model stages"
              title="The product surfaces the full ECG workflow"
              copy="From the raw signal to the explanation overlay, the frontend should reflect the same domain logic the Python code already uses."
            />

            <div className="mt-8 space-y-3">
              {modelStages.map((stage, index) => (
                <div key={stage} className="flex items-center gap-4 rounded-2xl border border-white/8 bg-white/5 p-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-sm font-semibold text-white">
                    0{index + 1}
                  </div>
                  <p className="text-sm font-medium text-slate-200">{stage}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6 sm:p-8">
            <SectionLabel
              eyebrow="Deliverable"
              title="A frontend that matches the project’s technical story"
              copy="The UI now communicates the architecture, the dataset status, the inference workflow, and the operational steps required to run the system."
            />

            <div className="mt-8 grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-white/8 bg-slate-950/45 p-5">
                <p className="text-sm text-slate-400">Primary mode</p>
                <p className="mt-2 text-lg font-semibold text-white">Operational dashboard</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Focuses on dataset readiness, model pipeline status, and ECG upload analysis.
                </p>
              </div>
              <div className="rounded-2xl border border-white/8 bg-slate-950/45 p-5">
                <p className="text-sm text-slate-400">Visual language</p>
                <p className="mt-2 text-lg font-semibold text-white">Clinical, modern, atmospheric</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Uses strong contrast, glass panels, gradients, and clear hierarchy.
                </p>
              </div>
            </div>
          </Card>
        </section>
      </div>
    </main>
  );
}

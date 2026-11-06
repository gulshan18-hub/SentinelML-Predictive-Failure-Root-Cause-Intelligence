import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  ArrowDown,
  ArrowUpRight,
  Bot,
  Check,
  CircleHelp,
  Database,
  Github,
  Layers3,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react';

type ApiStatus = 'checking' | 'online' | 'offline';

type HealthResponse = {
  status: string;
  service: string;
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
).replace(/\/+$/, '');
const REPOSITORY_URL =
  'https://github.com/gulshan18-hub/SentinelML-Predictive-Failure-Root-Cause-Intelligence';

const workflowSteps = [
  {
    number: '01',
    icon: Database,
    title: 'Prepare the dataset',
    description:
      'Download the official AI4I dataset, validate its schema, and generate an exploratory report.',
    command: 'python -m scripts.prepare_dataset --download',
    href: `${REPOSITORY_URL}/blob/main/docs/dataset.md`,
    linkLabel: 'Dataset guide',
    status: 'Ready',
  },
  {
    number: '02',
    icon: Bot,
    title: 'Train and evaluate',
    description:
      'Compare candidates against a baseline and evaluate the selected pipeline on a held-out test split.',
    command: 'python -m scripts.train_model',
    href: `${REPOSITORY_URL}/blob/main/docs/ml-methodology.md`,
    linkLabel: 'ML methodology',
    status: 'Ready',
  },
  {
    number: '03',
    icon: Layers3,
    title: 'Serve predictions',
    description:
      'A prediction-serving API and live machine telemetry integration are not implemented yet.',
    command: null,
    href: null,
    linkLabel: null,
    status: 'Not available',
  },
];

export default function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [apiError, setApiError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState('overview');
  const activeRequest = useRef<AbortController | null>(null);

  const refreshApiStatus = useCallback(async () => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setApiStatus('checking');
    setApiError(null);

    let timedOut = false;
    const timeout = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 5000);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const health = (await response.json()) as HealthResponse;
      if (health.status !== 'ok' || health.service !== 'sentinelml-backend') {
        throw new Error('Backend health response was unexpected');
      }

      setApiStatus('online');
    } catch (error) {
      if (activeRequest.current !== controller) {
        return;
      }

      setApiStatus('offline');
      setApiError(
        timedOut
          ? 'Request timed out. Check that the backend is running.'
          : error instanceof Error
            ? error.message
            : 'Unable to reach the backend API',
      );
    } finally {
      window.clearTimeout(timeout);
      if (activeRequest.current === controller) {
        activeRequest.current = null;
      }
    }
  }, []);

  useEffect(() => {
    void refreshApiStatus();
    const interval = window.setInterval(() => void refreshApiStatus(), 30_000);

    return () => {
      window.clearInterval(interval);
      const controller = activeRequest.current;
      activeRequest.current = null;
      controller?.abort();
    };
  }, [refreshApiStatus]);

  const statusLabel = {
    checking: 'Checking connection',
    online: 'API connected',
    offline: 'API unavailable',
  }[apiStatus];

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="SentinelML home">
          <span className="brand-mark">
            <Activity size={21} strokeWidth={2.2} />
          </span>
          <span className="brand-name">
            sentinel<span>ml</span>
          </span>
        </a>

        <div className="workspace-picker">
          <span className="workspace-avatar">S</span>
          <span className="workspace-copy">
            <span className="workspace-label">WORKSPACE</span>
            <span className="workspace-name">Development</span>
          </span>
          <span className="workspace-dot" aria-label="Development workspace" />
        </div>

        <p className="nav-heading">WORKSPACE</p>
        <nav className="primary-nav" aria-label="Primary navigation">
          <a
            className={`nav-link ${activeSection === 'overview' ? 'active' : ''}`}
            href="#overview"
            aria-current={activeSection === 'overview' ? 'page' : undefined}
            onClick={() => setActiveSection('overview')}
          >
            <Activity size={17} />
            <span>Overview</span>
          </a>
          <a
            className={`nav-link ${activeSection === 'pipeline' ? 'active' : ''}`}
            href="#pipeline"
            aria-current={activeSection === 'pipeline' ? 'page' : undefined}
            onClick={() => setActiveSection('pipeline')}
          >
            <Workflow size={17} />
            <span>ML pipeline</span>
          </a>
          <a
            className={`nav-link ${activeSection === 'resources' ? 'active' : ''}`}
            href="#resources"
            aria-current={activeSection === 'resources' ? 'page' : undefined}
            onClick={() => setActiveSection('resources')}
          >
            <CircleHelp size={17} />
            <span>Resources</span>
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="sidebar-note-icon">
              <ShieldCheck size={17} />
            </span>
            <p>
              <strong>Research workspace</strong>
              <span>Benchmark data only</span>
            </p>
          </div>
          <a
            className="sidebar-github"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
          >
            <Github size={16} />
            <span>View source</span>
            <ArrowUpRight size={14} />
          </a>
        </div>
      </aside>

      <main className="main-content" id="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>SentinelML</span>
            <span className="breadcrumb-divider">/</span>
            <span className="breadcrumb-current">Workspace overview</span>
          </div>
          <div className="topbar-actions">
            <span className="environment-badge">
              <span className="environment-dot" />
              Development
            </span>
            <a
              className="icon-link"
              href={REPOSITORY_URL}
              target="_blank"
              rel="noreferrer"
              aria-label="Open SentinelML on GitHub"
            >
              <Github size={18} />
            </a>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-section" id="overview">
            <div>
              <p className="eyebrow">
                <Sparkles size={14} />
                PREDICTIVE MAINTENANCE WORKSPACE
              </p>
              <h1>Make every model decision traceable.</h1>
              <p className="welcome-description">
                Prepare benchmark data, evaluate failure-event models, and inspect
                the lifecycle from one place.
              </p>
            </div>
            <a className="primary-button" href="#pipeline">
              Explore the pipeline
              <ArrowDown size={16} />
            </a>
          </section>

          <section className="status-grid" aria-label="Workspace status">
            <article className={`status-card api-card api-${apiStatus}`}>
              <div className="status-card-top">
                <span className="card-icon api-icon">
                  <Activity size={18} />
                </span>
                <span
                  className={`status-pill status-${apiStatus}`}
                  role="status"
                  aria-live="polite"
                >
                  <span className="status-indicator" />
                  {statusLabel}
                </span>
              </div>
              <p className="card-label">Backend service</p>
              <div className="card-value-row">
                <h2>{apiStatus === 'online' ? 'Connected' : apiStatus === 'offline' ? 'Offline' : 'Connecting'}</h2>
                <button
                  className="refresh-button"
                  type="button"
                  onClick={() => void refreshApiStatus()}
                  disabled={apiStatus === 'checking'}
                  aria-label="Refresh backend connection"
                  title="Refresh backend connection"
                >
                  <RefreshCw
                    size={15}
                    className={apiStatus === 'checking' ? 'spin' : ''}
                  />
                </button>
              </div>
              <p className="card-footnote" title={apiError ?? `${API_BASE_URL}/api/v1/health`}>
                {apiError ?? `${API_BASE_URL}/api/v1/health`}
              </p>
            </article>

            <article className="status-card">
              <div className="status-card-top">
                <span className="card-icon dataset-icon">
                  <Database size={18} />
                </span>
                <span className="quiet-badge">Benchmark</span>
              </div>
              <p className="card-label">Data source</p>
              <h2>AI4I 2020</h2>
              <p className="card-footnote">Synthetic dataset · not live telemetry</p>
            </article>

            <article className="status-card">
              <div className="status-card-top">
                <span className="card-icon model-icon">
                  <Bot size={18} />
                </span>
                <span className="quiet-badge">CLI workflow</span>
              </div>
              <p className="card-label">Model lifecycle</p>
              <h2>Train &amp; evaluate</h2>
              <p className="card-footnote">Baseline comparison and held-out evaluation</p>
            </article>

            <article className="status-card">
              <div className="status-card-top">
                <span className="card-icon serving-icon">
                  <Layers3 size={18} />
                </span>
                <span className="quiet-badge muted-badge">Not available</span>
              </div>
              <p className="card-label">Prediction serving</p>
              <h2>Not implemented</h2>
              <p className="card-footnote">No live inference endpoint or telemetry feed</p>
            </article>
          </section>

          <section className="pipeline-section" id="pipeline">
            <div className="section-heading">
              <div>
                <p className="eyebrow">YOUR WORKFLOW</p>
                <h2>From dataset to validated artifact</h2>
              </div>
              <span className="section-caption">Three clear steps</span>
            </div>

            <div className="workflow-list">
              {workflowSteps.map((step) => {
                const StepIcon = step.icon;

                return (
                  <article className="workflow-step" key={step.number}>
                    <div className="step-leading">
                      <span className="step-icon">
                        <StepIcon size={19} />
                      </span>
                      <span className="step-number">{step.number}</span>
                    </div>
                    <div className="step-content">
                      <div className="step-title-row">
                        <h3>{step.title}</h3>
                        <span
                          className={`step-status ${step.status === 'Ready' ? 'step-ready' : 'step-unavailable'}`}
                        >
                          {step.status === 'Ready' && <Check size={12} />}
                          {step.status}
                        </span>
                      </div>
                      <p>{step.description}</p>
                      {step.command && (
                        <code className="command-line">{step.command}</code>
                      )}
                    </div>
                    {step.href && step.linkLabel ? (
                      <a
                        className="step-link"
                        href={step.href}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {step.linkLabel}
                        <ArrowUpRight size={15} />
                      </a>
                    ) : (
                      <span className="step-link disabled-link">Coming later</span>
                    )}
                  </article>
                );
              })}
            </div>
          </section>

          <section className="notice-card" aria-label="Benchmark limitations">
            <span className="notice-icon">
              <ShieldCheck size={19} />
            </span>
            <div>
              <h2>Built for careful evaluation, not production claims</h2>
              <p>
                AI4I is synthetic and labels failures at the same recorded
                datapoint. It does not establish future failure prediction,
                real-factory performance, or calibrated probabilities.
              </p>
            </div>
            <a
              href={`${REPOSITORY_URL}/blob/main/docs/ml-methodology.md`}
              target="_blank"
              rel="noreferrer"
            >
              Read the methodology
              <ArrowUpRight size={15} />
            </a>
          </section>

          <section className="resources-section" id="resources">
            <div className="section-heading">
              <div>
                <p className="eyebrow">PROJECT DOCUMENTATION</p>
                <h2>Useful references</h2>
              </div>
            </div>
            <div className="resource-grid">
              <a
                className="resource-card"
                href={`${REPOSITORY_URL}/blob/main/docs/dataset.md`}
                target="_blank"
                rel="noreferrer"
              >
                <span className="resource-icon dataset-icon">
                  <Database size={18} />
                </span>
                <span className="resource-copy">
                  <strong>Dataset guide</strong>
                  <span>Source, schema, validation, and limitations</span>
                </span>
                <ArrowUpRight size={16} />
              </a>
              <a
                className="resource-card"
                href={`${REPOSITORY_URL}/blob/main/docs/ml-methodology.md`}
                target="_blank"
                rel="noreferrer"
              >
                <span className="resource-icon model-icon">
                  <Bot size={18} />
                </span>
                <span className="resource-copy">
                  <strong>ML methodology</strong>
                  <span>Splits, model selection, metrics, and artifacts</span>
                </span>
                <ArrowUpRight size={16} />
              </a>
            </div>
          </section>

          <footer className="page-footer">
            <span>SentinelML <span className="footer-version">· Development preview</span></span>
            <a href={REPOSITORY_URL} target="_blank" rel="noreferrer">
              Open source project
              <ArrowUpRight size={13} />
            </a>
          </footer>
        </div>
      </main>
    </div>
  );
}

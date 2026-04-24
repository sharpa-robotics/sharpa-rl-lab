# Contributing to sharpa-rl-lab

## nexus 설치

sharpa-rl-lab은 실험 로깅을 위해 [nexus](https://github.com/jonghochoi/nexus)를 사용합니다.
nexus는 `pyproject.toml`에 의존성으로 등록되어 있으며, 아래 두 가지 방법으로 설치할 수 있습니다.

### 방법 1 — 패키지로 설치 (권장)

sharpa-rl-lab을 설치할 때 nexus가 자동으로 함께 설치됩니다.

```bash
pip install -e ".[dev]"   # 또는
pip install -e .
```

nexus만 단독으로 설치하려면:

```bash
pip install git+https://github.com/jonghochoi/nexus.git
```

### 방법 2 — 로컬 editable 설치 (nexus 코드를 직접 수정하는 경우)

nexus와 sharpa-rl-lab을 같은 환경에서 함께 개발하는 경우:

```bash
# nexus를 editable 모드로 먼저 설치
pip install -e /path/to/nexus

# 이후 sharpa-rl-lab 설치
pip install -e .
```

로컬 nexus가 설치되어 있으면 `pyproject.toml`의 git URL 의존성보다 우선합니다.

---

## MLflow 서버 실행

`logger_mode`가 `"dual"` 또는 `"mlflow"`인 경우, 훈련 전에 로컬 MLflow 서버가 실행 중이어야 합니다.

```bash
# nexus 레포에서 제공하는 스크립트로 서버 시작 (백그라운드 유지)
bash /path/to/nexus/scheduled_sync/start_local_mlflow.sh

# 서버 확인: http://127.0.0.1:5100
```

서버 없이 시작하려면 config에서 `logger_mode: 'tensorboard'`로 설정하면 됩니다.

---

## 로거 설정

`rl_isaaclab/tasks/inhand_rotate/agents/ppo_cfg.yaml`의 `algorithm` 섹션에서 제어합니다.

```yaml
algorithm:
  experiment_name: 'my_experiment'       # MLflow experiment 이름
  logger_mode: 'dual'                    # 'dual' | 'mlflow' | 'tensorboard'
  mlflow_tracking_uri: 'http://127.0.0.1:5100'
```

| `logger_mode` | TensorBoard | MLflow | 용도 |
|:---:|:---:|:---:|---|
| `dual` | ✅ | ✅ | 기본값. 두 곳 모두 기록 |
| `mlflow` | ❌ | ✅ | MLflow만 사용 |
| `tensorboard` | ✅ | ❌ | MLflow 서버 없을 때 fallback |

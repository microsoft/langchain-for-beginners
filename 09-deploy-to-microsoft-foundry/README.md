# Deploy LangChain Agents to Microsoft Foundry

In this optional chapter, you'll take the LangChain agent you built in [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md) and deploy it to **Microsoft Foundry** as a hosted agent. You'll keep the same `create_agent()` pattern, configure model access, test the hosting protocol locally, and deploy the code to an existing Foundry project.

This chapter follows the [Microsoft Foundry hosted agent quickstart](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent). It assumes you already have a Foundry project and model deployment. If you don't, use the quickstart to create them, then return to Step 2.

## Prerequisites

- Completed [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)
- Completed the local setup in [Course Setup](../00-course-setup/README.md)
- A Foundry project with a deployed chat model, such as `gpt-4.1` or `gpt-5-mini`, and an embedding model, such as `text-embedding-3-small`
- **Foundry Project Manager** access on that project for hosted-agent deployment
- Python 3.13 or later for source-code hosted agent deployment
- [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) with the `azure.ai.agents` extension

> **Authentication**: `AI_API_KEY` configures model calls in the application, while `azd auth login` authenticates the deployment CLI in Step 5.

> **Use your existing virtual environment**: Before running the commands in this chapter, make sure the Python virtual environment you created in [Course Setup](../00-course-setup/README.md#step-2-clone-the-repository) is active. Do not create a separate environment for Chapter 09. In GitHub Codespaces, `.venv` is activated automatically. For local development, reactivate the environment from the repository root if needed:
>
> **Windows:** `venv\Scripts\activate`
>
> **macOS or Linux:** `source venv/bin/activate`

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Explain what a Microsoft Foundry hosted agent provides
- ✅ Wrap a LangChain `create_agent()` graph for hosted deployment
- ✅ Use embedding-based semantic search in a hosted agentic RAG tool
- ✅ Configure model authentication with a Foundry API key
- ✅ Configure and test the hosted agent locally
- ✅ Deploy directly to an existing Foundry project with `azd`
- ✅ Verify and troubleshoot the deployment

---

## 📌 What You'll Build

The sample embeds a few course notes into LangChain's `InMemoryVectorStore` and searches them by semantic similarity. Microsoft Foundry provides the managed runtime, hosted endpoint, deployment history, and logs for the deployed agent.

The chapter files are:

- **Project**: [`code/`](./code/)
- **Code**: [`code/main.py`](./code/main.py)
- **Hosted agent config**: [`code/azure.yaml`](./code/azure.yaml)
- **Deployment dependencies**: [`code/requirements.txt`](./code/requirements.txt)
- **Environment template**: [`.env.example`](../.env.example) in the repository root

The important hosting and authentication code is:

```python
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_azure_ai.agents.hosting import ResponsesHostServer


load_dotenv()


def build_chat_model():
    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )


def build_embeddings_model():
    return OpenAIEmbeddings(
        base_url=os.getenv("AI_ENDPOINT"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("AI_API_KEY"),
    )


def build_agent():
    vector_store = InMemoryVectorStore.from_documents(
        COURSE_NOTES,
        build_embeddings_model(),
    )

    @tool
    def search_course_notes(query: str) -> str:
        results = vector_store.similarity_search(query, k=3)
        return "\n\n".join(
            f"[{document.metadata['source']}]: {document.page_content}"
            for document in results
        )

    return create_agent(
        build_chat_model(),
        tools=[search_course_notes],
        system_prompt="You are a helpful agentic RAG assistant...",
    )


port = int(os.getenv("PORT", "8088"))
ResponsesHostServer(build_agent()).run(port=port)
```

`ChatOpenAI` and `OpenAIEmbeddings` both use `AI_ENDPOINT` and `AI_API_KEY` for model requests. The course notes are embedded when the agent starts, and `similarity_search()` retrieves the three notes closest to the tool query. `ResponsesHostServer` exposes `/responses` using the protocol Foundry uses to send input to the agent and receive its output.

---

## Step 1: Prepare Your LangChain Agent for Hosting

Open [`main.py`](./code/main.py) and find `search_course_notes`. It follows the Chapter 8 pattern: `InMemoryVectorStore.from_documents()` embeds `COURSE_NOTES`, and the tool runs `similarity_search()` when the agent decides retrieval is needed.

Run the sample unchanged first. After it works, you can add your own documents or replace `InMemoryVectorStore` with a persistent vector store. Your tools, document metadata, retrieval logic, and system prompt can change without changing `build_chat_model()` or the `ResponsesHostServer` wrapper.

---

## Step 2: Configure Your Foundry Project

Chapter 09 uses the same `.env` file in the repository root for local testing and deployment.

If you have not created the root `.env` yet, create it from [`.env.example`](../.env.example):

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**macOS or Linux:**

```bash
cp .env.example .env
```

Open the root `.env` and add these four Chapter 09 deployment values:

| Where to find the value                                           | Variables                                                        |
| ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| Foundry portal, project overview                                  | `FOUNDRY_PROJECT_ENDPOINT`                                       |
| Azure portal, the Foundry resource overview and project JSON view | `AZURE_SUBSCRIPTION_ID`, `AZURE_AI_PROJECT_ID`, `AZURE_LOCATION` |

`FOUNDRY_PROJECT_ENDPOINT` is the project endpoint ending in `/api/projects/<project>`. The repository ignores the root `.env`; `main.py` loads it for local testing, and Step 5 imports it into the azd environment.

---

## Step 3: Confirm Dependencies

No additional installation is required if you completed [Course Setup](../00-course-setup/README.md). The Chapter 09 [`code/requirements.txt`](./code/requirements.txt) matches the repository-root [`requirements.txt`](../requirements.txt) and is included so Foundry can install the same dependencies during deployment.

---

## Step 4: Test the Hosted Wrapper Locally

After completing the root `.env` file in Step 2, run the host from the source folder:

```bash
cd 09-deploy-to-microsoft-foundry/code
python main.py
```

The host listens on `http://localhost:8088/responses`. In another terminal, send a test request.

**PowerShell:**

```powershell
$body = @{
    input = "When was LangChain created?"
    stream = $false
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri http://localhost:8088/responses `
    -Method Post `
    -Body $body `
    -ContentType "application/json"

$response.output |
    Where-Object { $_.type -eq "message" } |
    ForEach-Object { $_.content } |
    Where-Object { $_.type -eq "output_text" } |
    ForEach-Object { $_.text }
```

**Bash:**

```bash
curl -sS -H "Content-Type: application/json" \
    -X POST http://localhost:8088/responses \
    -d '{"input":"When was LangChain created?","stream":false}'
```

If the response says LangChain was created in 2022, embedding search, model authentication, and the hosted Responses wrapper are working. Press **Ctrl+C** in the server terminal when you finish the local test.

---

## Step 5: Deploy to an Existing Foundry Project with `azd`

The [`code/`](./code/) folder is already an azd hosted-agent project. `AI_ENDPOINT` configures the running agent, but it does not tell `azd` where to deploy. For direct code deployment, azd separately requires the subscription ID, project endpoint, full resource ID, and Azure region collected in Step 2.

> **Existing project**: Do not run `azd provision` when the Foundry project and model deployments already exist. With `infra.provider: microsoft.foundry`, provisioning can create a new resource group, Foundry resource, and project. If `azd deploy` suggests running `azd provision`, first confirm that all four deployment values below are set; the generic suggestion can also mean that `AZURE_SUBSCRIPTION_ID` is missing.

Install the agent extension, sign in, and create a local azd environment:

```bash
cd 09-deploy-to-microsoft-foundry/code
azd ext install azure.ai.agents
azd auth login
azd env new langchain-course # Or other name you prefer
```

If the environment already exists, select it instead:

```bash
azd env select langchain-course # Or name of your existing environment
```

Load the deployment and runtime values directly from the root `.env` file into the active azd environment. Because you are now in the Chapter 09 `code` folder, the root file is two directories above it. Keep `/openai/v1` off `FOUNDRY_PROJECT_ENDPOINT`, and include the complete project resource ID in `AZURE_AI_PROJECT_ID`:

```bash
azd env set --file ../../.env
```

Verify that azd resolves the intended project before deploying:

```bash
azd ai project show --output json
azd env get-values
```

Check that `azd ai project show` prints the intended project endpoint without `/openai/v1` and resolves it from the active azd environment rather than the global config. In `azd env get-values`, confirm that `AZURE_SUBSCRIPTION_ID`, `FOUNDRY_PROJECT_ENDPOINT`, `AZURE_AI_PROJECT_ID`, and `AZURE_LOCATION` are all set. Then deploy the source directly. Because `codeConfiguration` is present in `azure.yaml`, Foundry performs a remote build using `code/requirements.txt`:

```bash
azd deploy --no-prompt
azd ai agent show --output json
azd ai agent invoke "Explain agentic RAG in one paragraph."
```

A successful deployment creates an immutable agent version. `azd ai agent show` should report an `active` or `deployed` status and list the Responses endpoint.

> **⚠️ Cleanup**: Hosted agents and model calls can incur Azure charges. Delete agents and resources that you created only for this exercise when you finish.

---

## 🔧 Troubleshooting

| Problem                                       | What to check                                                                                                                                                                                                                                                           |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `azd deploy` asks for `azd provision`         | For an existing project, first check `AZURE_SUBSCRIPTION_ID`. Also verify `FOUNDRY_PROJECT_ENDPOINT`, `AZURE_AI_PROJECT_ID`, and `AZURE_LOCATION`; do not run `azd provision` unless you intend to create or update infrastructure.                                     |
| `agent.yaml` or `agent.yml` does not exist    | The installed azd extensions might not support the inline agent configuration in `azure.yaml`. Run `azd extension upgrade --all`, then retry `azd deploy`.                                                                                                              |
| `AZURE_SUBSCRIPTION_ID` is not set            | Set it explicitly with `azd env set AZURE_SUBSCRIPTION_ID "<subscription-id>"`. azd does not derive it from the subscription segment inside `AZURE_AI_PROJECT_ID`.                                                                                                      |
| `no Foundry project endpoint resolved`        | Select the intended azd environment and set `FOUNDRY_PROJECT_ENDPOINT` without `/openai/v1`.                                                                                                                                                                            |
| `Microsoft Foundry project ID is required`    | Set `AZURE_AI_PROJECT_ID` to the project's full Azure Resource Manager ID, beginning with `/subscriptions/`.                                                                                                                                                            |
| `AZURE_LOCATION is not set`                   | Set `AZURE_LOCATION` to the Azure region of the existing Foundry resource, such as `swedencentral`.                                                                                                                                                                     |
| Model request returns `401` or `403`          | Confirm that `AI_API_KEY` and `AI_ENDPOINT` came from the same Foundry resource.                                                                                                                                                                                        |
| `DeploymentNotFound` or `ResourceNotFound`    | Check that `AI_MODEL` and `AI_EMBEDDING_MODEL` are deployment names and that `AI_ENDPOINT` belongs to the same Foundry resource.                                                                                                                                        |
| Embedding request returns `404`               | Confirm that `AI_ENDPOINT` ends in `/openai/v1` and that `AI_EMBEDDING_MODEL` names an embedding deployment on that resource.                                                                                                                                           |
| `Conversation '<id>' not found` returns `404` | `azd ai agent invoke` can reuse locally persisted session and conversation IDs after the remote state has expired or changed. Run `azd ai agent invoke --new-session --new-conversation "<message>"` once to reset both; subsequent invokes can use the normal command. |
| Local request reports connection refused      | Keep `python main.py` running and confirm that port `8088` is available.                                                                                                                                                                                                |
| A new deployment is temporarily not ready     | Wait for `azd ai agent show` to report `active` or `deployed`, then invoke it again.                                                                                                                                                                                    |

---

## 🎓 Key Takeaways

- You can deploy a LangChain `create_agent()` graph to Microsoft Foundry as a hosted agent
- `OpenAIEmbeddings` and `InMemoryVectorStore` provide semantic retrieval without changing the hosted-agent wrapper
- `ResponsesHostServer` exposes the graph through the Foundry Responses protocol
- `AI_ENDPOINT` and `AI_API_KEY` configure model access for both LangChain clients
- `AZURE_SUBSCRIPTION_ID`, `FOUNDRY_PROJECT_ENDPOINT`, `AZURE_AI_PROJECT_ID`, and `AZURE_LOCATION` identify the deployment target, while `AI_ENDPOINT` is used for model inference
- Chapter 09 uses the repository root `.env` for both local testing and `azd` configuration
- Existing projects can be targeted with `azd deploy` without running `azd provision`

---

## 🏆 Assignment

Ready to practice? Complete the deployment challenge in [assignment.md](./assignment.md)!

---

## 🗺️ Navigation

[← Previous: Agentic RAG Systems](../08-agentic-rag-systems/README.md) | [Back to Main](../README.md)

---

## 🎉 Congratulations!

You've completed the LangChain for Beginners course and learned how to take a local LangChain agent all the way to a Microsoft Foundry hosted agent.

---

## 💬 Questions?

[![Microsoft Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)

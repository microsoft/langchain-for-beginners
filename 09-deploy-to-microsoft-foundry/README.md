# Deploy LangChain Agents to Microsoft Foundry

In this optional chapter, you'll take the LangChain agent you built in [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md) and deploy it to **Microsoft Foundry** as a hosted agent. You'll keep the same `create_agent()` pattern, add keyless authentication, test the hosting protocol locally, and deploy the code to an existing Foundry project.

This chapter follows the [Microsoft Foundry hosted agent quickstart](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent?pivots=vscode). It assumes you already have a Foundry project and model deployment. If you don't, use the quickstart or Foundry Toolkit to create them, then return to Step 2.

## Prerequisites

- Completed [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)
- Completed the local setup in [Course Setup](../00-course-setup/README.md)
- A Foundry project with a deployed chat model, such as `gpt-4.1` or `gpt-5-mini`
- **Foundry Project Manager** access on that project for hosted-agent deployment
- Python 3.13 or later for source-code hosted agent deployment
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) for local authentication
- [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) with the `azure.ai.agents` extension
- Optional: VS Code with the [Microsoft Foundry Toolkit](https://aka.ms/foundrytk) for guided deployment

> **Authentication**: This chapter uses two sign-in commands for different tools. `az login` authenticates the local Python application in Step 4, while `azd auth login` authenticates the deployment CLI in Step 5.

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Explain what a Microsoft Foundry hosted agent provides
- ✅ Wrap a LangChain `create_agent()` graph for hosted deployment
- ✅ Authenticate without storing an API key
- ✅ Configure and test the hosted agent locally
- ✅ Deploy directly to an existing Foundry project with `azd`
- ✅ Verify and troubleshoot the deployment

---

## 📌 What You'll Build

The sample uses a few in-memory course notes instead of a vector store so the deployment pattern stays visible. Microsoft Foundry provides the managed runtime, hosted endpoint, deployment history, logs, and a dedicated identity for the deployed agent.

The chapter files are:

- **Project**: [`code/`](./code/)
- **Code**: [`code/main.py`](./code/main.py)
- **Hosted agent config**: [`code/azure.yaml`](./code/azure.yaml)
- **Dependencies**: [`code/requirements.txt`](./code/requirements.txt)
- **Environment template**: [`code/.env.example`](./code/.env.example)

The important hosting and authentication code is:

```python
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_azure_ai.agents.hosting import ResponsesHostServer


def build_chat_model():
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://ai.azure.com/.default",
    )
    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=token_provider,
    )


def build_agent():
    return create_agent(
        build_chat_model(),
        tools=[search_course_notes],
        system_prompt="You are a helpful agentic RAG assistant...",
    )


port = int(os.getenv("PORT", "8088"))
ResponsesHostServer(build_agent()).run(port=port)
```

`ChatOpenAI` accepts the token provider through its OpenAI-compatible `api_key` parameter, but the callable returns short-lived Microsoft Entra tokens rather than a stored API key. `ResponsesHostServer` exposes `/responses` using the protocol Foundry uses to send input to the agent and receive its output.

---

## Step 1: Prepare Your LangChain Agent for Hosting

Open [`main.py`](./code/main.py) and find `search_course_notes`. This tool is intentionally small so you can focus on deployment first.

Run the sample unchanged first. After it works, you can replace `COURSE_NOTES` and `search_course_notes` with the vector store and retrieval tool from Chapter 8. Your tools, document metadata, retrieval logic, and system prompt can change without changing `build_chat_model()` or the `ResponsesHostServer` wrapper.

---

## Step 2: Configure Your Foundry Project

Open your project in the Foundry portal. From the project overview and deployment pages, collect these values:

| Value | Variable | Example | Used by |
| --- | --- | --- | --- |
| Project endpoint | `FOUNDRY_PROJECT_ENDPOINT` | `https://<resource>.services.ai.azure.com/api/projects/<project>` | `azd` when selecting the deployment target |
| Project resource ID | `AZURE_AI_PROJECT_ID` | `/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource>/projects/<project>` | `azd` when identifying the existing project |
| Azure region | `AZURE_LOCATION` | `swedencentral` | `azd` when building and deploying the agent code |
| Model inference endpoint | `AI_ENDPOINT` | `https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1` | The running agent when calling the model |
| Chat deployment name | `AI_MODEL` | `gpt-4.1` | `ChatOpenAI`; use the deployment name, not the model catalog ID |
| Embedding deployment name | `AI_EMBEDDING_MODEL` | `text-embedding-3-small` | Your Chapter 8 vector-store adaptation |

`AI_ENDPOINT` is the project endpoint with `/openai/v1` appended. Do not use the inference endpoint as `FOUNDRY_PROJECT_ENDPOINT`.

The project resource ID is the full Azure Resource Manager ID, not the project endpoint or project name. You can copy it from the project's JSON view in the Azure portal. Use the region of the parent Foundry resource for `AZURE_LOCATION`.

Copy [`code/.env.example`](./code/.env.example) to a local `.env` file inside `code/`, then replace the placeholders. The local application does not need the three azd deployment values; you will store them in the azd environment in Step 5.

```bash
AI_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1
AI_MODEL=<your-chat-model-deployment-name>
AI_EMBEDDING_MODEL=<your-embedding-model-deployment-name>
```

The current in-memory search does not call the embedding model, but the variable is ready for the Chapter 8 adaptation. The repository ignores `.env`, and it should contain only these non-secret values.

Do not add `AI_API_KEY`. `DefaultAzureCredential` uses your Azure CLI identity locally and the dedicated Microsoft Entra agent identity after deployment. Foundry grants the hosted identity access to model inference through its project endpoint, so the code does not need separate production credentials.

---

## Step 3: Install Hosted-Agent Dependencies

Use the same virtual environment style from [Course Setup](../00-course-setup/README.md). From the repository root, activate your existing course environment and install the hosted-agent sample requirements.

**Windows PowerShell:**

```powershell
venv\Scripts\activate
pip install -r 09-deploy-to-microsoft-foundry/code/requirements.txt
```

**macOS or Linux:**

```bash
source venv/bin/activate
pip install -r 09-deploy-to-microsoft-foundry/code/requirements.txt
```

If you are using GitHub Codespaces, the course environment is already activated; run the same `pip install -r ...` command.

---

## Step 4: Test the Hosted Wrapper Locally

Sign in with Azure CLI so `DefaultAzureCredential` can access your Foundry project:

```bash
az login
az account show --output table
```

If you have access to more than one tenant or subscription, make sure the displayed account contains the Foundry project.

Then run the host from the source folder:

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

If the response says LangChain was created in 2022, both model authentication and the hosted Responses wrapper are working. Press **Ctrl+C** in the server terminal when you finish the local test.

---

## Step 5: Deploy to an Existing Foundry Project with `azd`

The [`code/`](./code/) folder is already an azd hosted-agent project. `AI_ENDPOINT` configures the running agent, but it does not tell `azd` where to deploy. For direct code deployment, azd separately requires the project endpoint, full resource ID, and Azure region collected in Step 2.

> **Existing project**: Do not run `azd provision` when the Foundry project and model deployments already exist. `azd provision` intentionally creates or updates Azure infrastructure; the steps below bind `azd deploy` directly to your existing project.

Install the agent extension, sign in, and create a local azd environment:

```bash
cd 09-deploy-to-microsoft-foundry/code
azd ext install azure.ai.agents
azd auth login
azd env new <environment-name>
```

If the environment already exists, select it instead:

```bash
azd env select <environment-name>
```

Set the three deployment values from Step 2, followed by the runtime values from your `.env` file. Keep `/openai/v1` off `FOUNDRY_PROJECT_ENDPOINT`, and include the complete project resource ID in `AZURE_AI_PROJECT_ID`:

```bash
azd env set FOUNDRY_PROJECT_ENDPOINT "https://<resource>.services.ai.azure.com/api/projects/<project>"
azd env set AZURE_AI_PROJECT_ID "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource>/projects/<project>"
azd env set AZURE_LOCATION "<project-region>"
azd env set AI_ENDPOINT "https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1"
azd env set AI_MODEL "<your-chat-model-deployment-name>"
azd env set AI_EMBEDDING_MODEL "<your-embedding-model-deployment-name>"
```

Verify that azd resolves the intended project before deploying:

```bash
azd ai project show --output json
azd env get-values
```

Check that `azd ai project show` prints the intended project endpoint without `/openai/v1`. In `azd env get-values`, confirm that `AZURE_AI_PROJECT_ID` is the complete project resource ID and `AZURE_LOCATION` is the project's region. Then deploy the source directly. Because `codeConfiguration` is present in `azure.yaml`, Foundry performs a remote build from `requirements.txt`:

```bash
azd deploy --no-prompt
azd ai agent show --output json
azd ai agent invoke "Explain agentic RAG in one paragraph."
```

A successful deployment creates an immutable agent version. `azd ai agent show` should report an `active` or `deployed` status and list the Responses endpoint.

---

## Step 6: Alternative Guided Deployment with Foundry Toolkit

To deploy from VS Code instead of the CLI:

1. Open `09-deploy-to-microsoft-foundry/code` as the active folder.
2. Run **Foundry Toolkit: Deploy Hosted Agent** from the Command Palette.
3. Choose **Code** and then **Remote** so Foundry installs `requirements.txt` during the build.
4. Select the existing Foundry project, confirm the agent settings, and deploy.
5. Open **Hosted Agents (Preview)** and test the deployed agent in the **Playground**.

> **⚠️ Cleanup**: Hosted agents and model calls can incur Azure charges. Delete agents and resources that you created only for this exercise when you finish.

---

## 🔧 Troubleshooting

| Problem | What to check |
| --- | --- |
| `no Foundry project endpoint resolved` | Select the intended azd environment and set `FOUNDRY_PROJECT_ENDPOINT` without `/openai/v1`. |
| `Microsoft Foundry project ID is required` | Set `AZURE_AI_PROJECT_ID` to the project's full Azure Resource Manager ID, beginning with `/subscriptions/`. |
| `AZURE_LOCATION is not set` | Set `AZURE_LOCATION` to the Azure region of the existing Foundry resource, such as `swedencentral`. |
| `DefaultAzureCredential` fails locally | Run `az login`, verify `az account show`, and confirm that your account can access the project. |
| Model request returns `401` or `403` | Confirm your user can invoke the deployed model. The hosted agent uses its Foundry-provided identity, but your local test uses your own identity. |
| `DeploymentNotFound` or `ResourceNotFound` | Check that `AI_MODEL` is the deployment name and that `AI_ENDPOINT` belongs to the same project. |
| Local request reports connection refused | Keep `python main.py` running and confirm that port `8088` is available. |
| A new deployment is temporarily not ready | Wait for `azd ai agent show` to report `active` or `deployed`, then invoke it again. |

---

## 🎓 Key Takeaways

- You can deploy a LangChain `create_agent()` graph to Microsoft Foundry as a hosted agent
- `ResponsesHostServer` exposes the graph through the Foundry Responses protocol
- `DefaultAzureCredential` switches from your local Azure identity to the hosted agent identity without storing an API key
- `FOUNDRY_PROJECT_ENDPOINT`, `AZURE_AI_PROJECT_ID`, and `AZURE_LOCATION` identify the deployment target, while `AI_ENDPOINT` is used for model inference
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
# Deploy LangChain Agents to Microsoft Foundry

In this optional chapter, you'll take the LangChain agent you built in [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md) and deploy it to **Microsoft Foundry** as a hosted agent. You have already learned how to build agents that decide when to search documents. Now you'll learn how to host that agent so it can run in a managed cloud environment with a stable endpoint, deployment history, and Foundry tooling.

This chapter follows the [Microsoft Foundry hosted agent quickstart](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent?pivots=vscode), with one course-specific goal: keep your LangChain `create_agent()` logic and wrap it for Foundry hosting.

## Prerequisites

- Completed [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)
- Completed the local setup in [Course Setup](../00-course-setup/README.md)
- VS Code with the [Microsoft Foundry Toolkit](https://aka.ms/foundrytk)
- Azure subscription access
- Python 3.13 or later for source-code hosted agent deployment
- A Foundry project with a deployed chat model, such as `gpt-4.1` or `gpt-5-mini`

> **📝 Note**: The Microsoft quickstart shows how to create a new Foundry project. If your team already has one, you can select that existing Foundry project in VS Code instead. You need **Foundry Project Manager** access on the project.

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Explain what a Microsoft Foundry hosted agent provides
- ✅ Prepare a LangChain `create_agent()` graph for hosted deployment
- ✅ Configure a Foundry project and model deployment for your agent
- ✅ Test the hosted-agent wrapper locally
- ✅ Deploy the agent from VS Code with Microsoft Foundry Toolkit
- ✅ Understand how the hosted-agent YAML maps to the ContainerAgent schema

---

## 📌 About the Code Example

The sample project in this chapter is intentionally small. It uses a few in-memory course notes instead of a full vector store so you can focus on the hosting pattern first.

The same pattern works with the vector store and retrieval tools you built in Chapter 8:

1. Keep your LangChain model setup, tools, and system prompt.
2. Build the agent with `create_agent()`.
3. Wrap that graph with the Foundry Responses host.
4. Deploy the project with Foundry Toolkit.

---

## 🚀 Why Deploy to Microsoft Foundry?

Running locally is perfect while you're learning and iterating. Hosting is the next step when you want other people or applications to use your agent reliably.

Microsoft Foundry hosted agents give you:

- A managed runtime for your agent code
- A stable hosted endpoint
- Conversation sessions and Foundry agent tooling
- Deployment history and cloud logs
- Integration with Foundry projects, model deployments, and identity

Think of this chapter as the bridge from "I built a LangChain agent" to "my team can call this agent from a real application."

---

## 💻 Example: Host a LangChain Agentic RAG Agent

The sample project is here:

- **Project**: [`code/`](./code/)
- **Code**: [`code/main.py`](./code/main.py)
- **Hosted agent config**: [`code/azure.yaml`](./code/azure.yaml)
- **Dependencies**: [`code/requirements.txt`](./code/requirements.txt)

**Key code you'll work with:**

```python
from langchain.agents import create_agent
from langchain_azure_ai.agents.hosting import ResponsesHostServer


def build_agent():
    return create_agent(
        build_chat_model(),
        tools=[search_course_notes],
        system_prompt="You are a helpful agentic RAG assistant...",
    )


ResponsesHostServer(build_agent()).run(port=port)
```

`create_agent()` gives you the same LangChain pattern from Chapter 8. `ResponsesHostServer` exposes that graph through the Foundry Hosted Agent Responses protocol at `/responses`.

> **🤖 Try with [GitHub Copilot](../docs/copilot.md) Chat:** Want to adapt your Chapter 8 agent? Open the sample `main.py` and ask Copilot:
> - "Replace the in-memory course notes with the vector store from Chapter 8."
> - "Explain how ResponsesHostServer changes the way this LangChain agent is served."

---

## Step 1: Prepare Your LangChain Agent for Hosting

Open [`main.py`](./code/main.py) and find `search_course_notes`. This tool is intentionally small so you can focus on deployment first.

To deploy your own agentic RAG system, replace `COURSE_NOTES` and `search_course_notes` with the vector store and retrieval tool you built in Chapter 8.

The hosting pattern stays the same:

```python
def build_agent():
    return create_agent(
        build_chat_model(),
        tools=[search_course_notes],
        system_prompt="You are a helpful agentic RAG assistant...",
    )
```

Your agent can keep using LangChain tools, document metadata, retrieval logic, and system prompts. The only new responsibility is starting a host server that Foundry can call.

---

## Step 2: Configure Your Foundry Project

In VS Code, open the Command Palette (**Ctrl+Shift+P**) and use the Foundry Toolkit flow from the quickstart:

1. Select **Foundry Toolkit: Create Project** if you need a new Foundry project, or select your existing project in the Foundry Toolkit explorer.
2. Select **Foundry Toolkit: Open Model Catalog** and deploy a chat model if your project doesn't already have one.
3. Copy the OpenAI-compatible endpoint for your Foundry resource. It looks like:

   ```text
    https://<resource>.services.ai.azure.com/openai/v1
   ```

4. Copy the chat model deployment name and embedding model deployment name, for example `gpt-4.1` and `text-embedding-3-small`.

For local testing, add these non-secret values to your shell or to a local `.env` file inside `code/`:

```bash
AI_ENDPOINT=https://<resource>.services.ai.azure.com/openai/v1
AI_MODEL=<your-chat-model-deployment-name>
AI_EMBEDDING_MODEL=<your-embedding-model-deployment-name>
```

Do not add `AI_API_KEY`. The sample uses `DefaultAzureCredential` to request short-lived Microsoft Entra tokens instead of placing a long-lived shared secret in `.env` or the hosted deployment. Locally, `DefaultAzureCredential` uses your Azure CLI sign-in. When deployed, Foundry creates a dedicated Microsoft Entra identity for the hosted agent and the same code uses that identity without any credential changes.

The sample [`azure.yaml`](./code/azure.yaml) follows the ContainerAgent schema and passes only `AI_ENDPOINT`, `AI_MODEL`, and `AI_EMBEDDING_MODEL` into the hosted runtime. The code reads those same non-secret variable names for local testing and Foundry hosting.

---

## Step 3: Install Optional Hosted-Agent Dependencies

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
```

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

If this works locally, your LangChain graph is speaking the same protocol it will use in Foundry.

---

## Step 5: Deploy with Foundry Toolkit

Deploy with the guided VS Code flow from the quickstart:

1. Open `09-deploy-to-microsoft-foundry/code` as the active folder in VS Code.
2. Open the Command Palette and select **Foundry Toolkit: Deploy Hosted Agent**.
3. Choose **Code** as the deployment method.
4. Choose **Remote** as the package mode so Foundry builds dependencies from `requirements.txt`.
5. Confirm the agent name, project, API version, model deployment, and review page.
6. Select **Deploy**.

When deployment finishes, open **Hosted Agents (Preview)** in the Foundry Toolkit explorer, select your agent, and use the **Playground** tab to ask the same test question.

---

## Optional: Add Azure Developer CLI Automation Later

This sample's [`azure.yaml`](./code/azure.yaml) follows the **ContainerAgent** schema used by Microsoft Foundry Toolkit. It is not an Azure Developer CLI project file, so don't run `azd env new` or `azd deploy` directly from `09-deploy-to-microsoft-foundry/code`.

Because the [`code/`](./code/) folder already contains the hosted-agent source and ContainerAgent YAML, you don't need to generate another azd agent scaffold for this chapter. Use the Foundry Toolkit deployment flow in Step 5 directly from `09-deploy-to-microsoft-foundry/code`.

If your team wants repeatable command-line deployment later, treat that as a separate automation project outside this tutorial folder. Install the Azure Developer CLI and the agent extension there, then follow the Azure Developer CLI hosted-agent documentation for creating a wrapper project around this existing source folder:

```bash
azd ext install azure.ai.agents
azd auth login
```

In that separate azd wrapper project, set the OpenAI-compatible endpoint, chat model, and embedding model in the azd environment. Authentication still comes from `DefaultAzureCredential`: use your Azure sign-in locally and the hosted agent identity after deployment. Then use the normal local and deploy commands from the wrapper project root:

```bash
azd env set AI_ENDPOINT "https://<resource>.services.ai.azure.com/openai/v1"
azd env set AI_MODEL "<your-model-deployment-name>"
azd env set AI_EMBEDDING_MODEL "<your-embedding-model-deployment-name>"

azd ai agent run
azd ai agent invoke --local "When was LangChain created?"
azd deploy
azd ai agent invoke "Explain agentic RAG in one paragraph."
```

> **⚠️ Cleanup**: Deployed hosted agents and model calls can incur Azure charges. If you created a new resource group just for this optional chapter, delete it from the Azure portal or run `azd down` from the hosted-agent project after you confirm it contains only resources you want to remove.

---

## 🎓 Key Takeaways

- You can deploy a LangChain `create_agent()` graph to Microsoft Foundry as a hosted agent
- Foundry Toolkit lets you create a new Foundry project or select an existing one
- `ResponsesHostServer` exposes the LangChain graph through the Foundry Responses protocol
- Keep using the course virtual environment locally; Foundry builds hosted dependencies from `requirements.txt`
- Use Foundry Toolkit directly with the existing `code/` folder; create separate azd automation only if your team needs it later

---

## 📦 Dependencies

The hosted-agent sample has its own dependency file because Foundry Toolkit uses this folder as the deployment package and installs from `code/requirements.txt` during remote build. It includes the course LangChain packages plus the hosted-agent runtime dependencies needed by `main.py`.

```bash
pip install -r 09-deploy-to-microsoft-foundry/code/requirements.txt
```

---

## 🔧 Environment Variables

For local testing with Foundry, set only the non-secret model configuration:

```bash
AI_ENDPOINT=your-openai-compatible-endpoint
AI_MODEL=your-chat-model-deployment-name
AI_EMBEDDING_MODEL=your-embedding-model-deployment-name
```

Authentication is handled by `DefaultAzureCredential`, so no API key environment variable is required. Run `az login` before local testing; the deployed agent uses its Foundry-provided Microsoft Entra identity.

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
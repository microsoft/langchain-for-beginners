# Assignment: Deploy a LangChain Agent to Microsoft Foundry

## Overview

Adapt the agentic RAG system you built in Chapter 8 and deploy it to an existing Microsoft Foundry project as a hosted agent.

## Prerequisites

- Complete the steps and prerequisites in this [chapter](./README.md)
- Complete [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)

---

## Challenge: Deploy Your Agentic RAG Agent

**Goal**: Deploy the agentic RAG agent you built in Chapter 8 as a Microsoft Foundry hosted agent.

**Tasks**:

1. In [`code/main.py`](./code/main.py), replace the sample course notes and retrieval tool with your own Chapter 8 documents and retrieval logic.
2. Keep the `create_agent()` and `ResponsesHostServer` patterns, then update the system prompt for your knowledge base.
3. Add the four Chapter 09 deployment values to the root `.env`, using the [root `.env.example`](../.env.example) as a reference. Use this file for both local testing and deployment.
4. Follow Step 4 in the [chapter README](./README.md#step-4-test-the-hosted-wrapper-locally) to test a non-streaming request at `http://localhost:8088/responses`.
5. Follow Step 5 in the [chapter README](./README.md#step-5-deploy-to-an-existing-foundry-project-with-azd) to import the `.env` values, deploy with `azd deploy --no-prompt`, and verify the deployment with `azd ai agent show --output json`.
6. Use `azd ai agent invoke` to ask at least three questions:
   - One general question the agent can answer directly
   - One document-specific question that should use retrieval
   - One question that is not covered by your documents
7. Delete any agents or resources created only for this exercise when you finish.

**Success Criteria**:

- Agent starts locally without errors
- `/responses` accepts a non-streaming test request
- `azd ai agent show` reports the hosted agent as `active` or `deployed`
- Agent uses retrieval and cites a source for knowledge-base questions
- Agent answers gracefully when the knowledge base does not contain an answer
- Agent is deployed to the intended existing Foundry project without running `azd provision`

---

## Submission Checklist

Before completing this chapter:

- [ ] Chapter 8 documents and retrieval logic are adapted in `code/main.py`
- [ ] All four Chapter 09 deployment values are filled in within the root `.env`
- [ ] Local hosted wrapper runs successfully
- [ ] `azd ai agent show` reports an `active` or `deployed` agent
- [ ] The deployed agent answers all three test-question types appropriately
- [ ] Exercise-only agents and resources are cleaned up

---

## Need Help?

- **Chapter instructions and troubleshooting**: Review the [Chapter 09 README](./README.md)
- **Hosted agent quickstart**: Review the [Microsoft Foundry hosted agent quickstart](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent)
- **Agentic RAG basics**: Review [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)
- **Course setup**: Review [Course Setup](../00-course-setup/README.md)
- **Still stuck**: Join our [Discord community](https://aka.ms/foundry/discord)

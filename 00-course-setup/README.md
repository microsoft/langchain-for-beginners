# Course Setup

Welcome! Before we dive into building AI applications with LangChain, let's get your development environment ready. This chapter walks you through setting up your development environment and getting your GitHub token for free AI model access (you can also use Microsoft Foundry.) By the end, you'll have everything you need to start building with LangChain.

## Prerequisites

- A GitHub account (free)
- Basic command line knowledge
- Text editor or IDE

---

## 📖 The Workshop Analogy

**Just like setting up a workshop before building furniture, you need to prepare your development environment before building AI applications.**

You'll setup your development environment, get access to AI models, and configure your tools. This ensures you have a solid foundation and smooth development experience. Let's get your workshop ready—it takes just 15 minutes!

## Acessing Models: Github Models vs Microsoft Foundry

For this coures you can use GitHub Models or Microsoft Foundry to access LLMs. To get started we recommend using Github Models.

### Why GitHub Models?

- ✅ **Free**: No credit card required
- ✅ **Powerful**: Access to GPT-5, GPT-5-mini, and other models
- ✅ **Easy**: Use your existing GitHub account
- ✅ **Learning**: Perfect for this course!

While GitHub Models is a great option for this course, if you have an Azure subscription, you can use Microsoft Foundry for production-grade AI applications with enterprise features. If you are new to Azure you can also get a FREE Azure account with USD$200, by signing up [here](https://azure.microsoft.com/pricing/purchase-options/azure-account/?cid=msft_devrel-yt).

---

## Setup Options

Choose from one of the following options to set up your development environment. The easiest way to get started is GitHub Codespaces, since it will setup all the tools for you, but you can also set it up locally:

1. [**GitHub Codespaces**](#github-codespaces): Use a cloud-based development environment (**recommended**)
2. [**Local Development**](#local-development): Set up your environment on your machine.

---

## GitHub Codespaces

### Step 1: Open the Codespace for this course

You can open this course in GitHub Codespaces. The button below will take you to a page that has a green button that says `CREATE A CODESPACE`. Click that button and wait for the codespace to load. A web-based VS Code instance will open in your browser when loaded. 

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/microsoft/langchain-for-beginners)

The terminal should automatically open and in it you should see `(.venv)` in front of your username. This shows that a Python virtual enviroment has been activated. This environemnt already has all the python packages required for this course installed. 

---

### Step 2: Examine your `.env` file

All of your environment variables are automatically setup for you in Codespaces. 
GitHub Models is used by default and provides free access to AI models. To use Github Models you need a Personal Access Token. In Codespaces this token, the correct endpoint and model will automatically be added to your `.env` file.

Ensure that your `.env` file exists and examine it to make sure it has these required variables.

**For GitHub Models (Free):**

```bash
AI_API_KEY=your_personal_token_will_show_up_here
AI_ENDPOINT=https://models.inference.ai.azure.com
AI_MODEL=gpt-5-mini
```

**Alternative: Microsoft Foundry:**

If you have an Azure subscription, you can use Microsoft Foundry for production-grade AI applications and more tokens. See the [Azure Setup Appendix](./APPENDIX.md#azure-ai-foundry-setup) for detailed instructions on deploying models and configuring your environment.

Thats it🎉 You can now move on to [testing your setup](#test-your-setup).

---

## Local Development

### Step 1: Install Python

You'll need **Python 3.10 or higher** to run LangChain v1 applications.

#### Check if Python is installed

```bash
python --version
# or
python3 --version
```

If you see Python 3.10 or higher, you're good! Skip to [Step 2](#step-2-clone-the-repository). If not:

#### Install Python

1. Visit [python.org](https://www.python.org/downloads/)
2. Download Python 3.10+ for your operating system
3. Follow the installation instructions (make sure to check "Add Python to PATH" on Windows)
4. Verify installation:

```bash
python --version  # Should show 3.10 or higher
pip --version     # Should show pip version
```

---

### Step 2: Clone the Repository

```bash
# Clone the course repository
git clone https://github.com/microsoft/langchain-for-beginners

# Navigate to the project
cd langchain-for-beginners

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

This will install all required packages including:

- `langchain-openai` - OpenAI-compatible model integration
- `langchain-core` - Core LangChain functionality
- `langchain` - Main LangChain package with additional utilities
- `langchain-azure-ai` - Azure specific langchain integrations
- `python-dotenv` - Environment variable management for API keys

---

### Step 3: Create GitHub Personal Access Token

GitHub Models provides free access to powerful AI models—you just need a Personal Access Token.

#### Create Your Token

1. **Visit**: https://github.com/settings/tokens/new
2. **Token name**: `langchain-course` (or any name you prefer)
3. **Expiration**: Choose your preference (90 days recommended for learning)
4. **Scopes/Permissions**:
   - ✅ No scopes needed for GitHub Models!
   - You can leave all checkboxes unchecked
5. **Click**: "Generate token"
6. **⚠️ IMPORTANT**: Copy your token now and save it to a text file temporarily! You'll need it in the next step.

### Step 4: Configure Environment Variables

#### Create `.env` file

**Mac, Linux, WSL on Windows:**

```bash
cp .env.example .env
```

**Windows Command Prompt:**

```bash
# Windows Command Prompt
copy .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

#### Edit `.env` file

Open `.env` in your text editor and configure your AI provider.

**For GitHub Models (Free):**

```bash
AI_API_KEY=ghp_your_github_token_here
AI_ENDPOINT=https://models.inference.ai.azure.com
AI_MODEL=gpt-5-mini
```

**Replace `ghp_your_github_token_here` with your actual GitHub token!**

**Alternative: Microsoft Foundry:**

If you have an Azure subscription, you can use Microsoft Foundry for production-grade AI applications. See the [Azure Setup Appendix](./APPENDIX.md#azure-ai-foundry-setup) for detailed instructions on deploying models and configuring your environment.

## Test Your Setup

Let's verify everything works!

### Run the test

Run the following command in your terminal from the root of the project:

```bash
python scripts/test_setup.py
```

**Expected output:**

```bash
🚀 Testing AI provider connection...

✅ SUCCESS! Your AI provider is working!
   Provider: https://models.inference.ai.azure.com
   Model: gpt-5-mini

Model response: Setup successful!

🎉 You're ready to start the course!
```

If you see this, you're all set! If not, check the troubleshooting section below.

You may recieve the following 

```bash
❌ ERROR: Error code: 400 - {'error': {'code': 'unavailable_model', 'message': 'Unavailable model: gpt-5', 'details': 'Unavailable model: gpt-5'}}
```
Change the model to get round this you can see the full list at - (https://github.com/marketplace/models)

---

## ✅ Setup Checklist

Before starting the course, make sure you have:

For Codespaces:

- [ ] The Codepsace and terminal open
- [ ] A `.env` with the expected variables
- [ ] Test script runs successfully

For Local development:

- [ ] Python 3.10+ installed
- [ ] Project cloned and virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GitHub Personal Access Token created if you're using GitHub Models. If you're using Microsoft Foundry, ensure your models are deployed and you have your API key and endpoint.
- [ ] `.env` file configured with your token (or key if using Microsoft Foundry) and endpoint
- [ ] Test script runs successfully
- [ ] VS Code installed (optional but recommended)

---

## 🎯 What's Next?

You're all set! Time to build your first AI application.

**👉 Continue to [Introduction to LangChain](../01-introduction/README.md)**

---

## 📚 Additional Resources

- [GitHub Models Documentation](https://github.com/marketplace/models)
- [Python Documentation](https://docs.python.org/3/)
- [LangChain Python Documentation](https://python.langchain.com/)

---

## 🗺️ Navigation

[Back to Main](../README.md) | [Next: Introduction to LangChain →](../01-introduction/README.md)

---

## 💬 Questions?

If you get stuck or have any questions about building AI apps, join:

[![Microsoft Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)

If you have product feedback or errors while building visit:

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Azure_AI_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

If you run into issues with the course materials, please open an issue in the GitHub repo:

[![Course Issues](https://img.shields.io/badge/GitHub-LangChain_for_Beginners_Issues-blue?style=for-the-badge&logo=github&color=green&logoColor=fff)](https://github.com/microsoft/langchain-for-beginners/issues)

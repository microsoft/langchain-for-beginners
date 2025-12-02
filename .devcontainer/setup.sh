#!/bin/bash
set -e

echo "Setting up LangChain development environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create .env file from .env.example
echo "🔧 Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    
    # Replace AI_API_KEY with GitHub token if available
    if [ ! -z "$GITHUB_TOKEN" ]; then
        echo "🔑 Configuring AI_API_KEY with GitHub token..."
        sed -i "s/AI_API_KEY=your_github_personal_access_token/AI_API_KEY=$GITHUB_TOKEN/" .env
    else
        echo "⚠️  GITHUB_TOKEN not found. Please update AI_API_KEY in .env manually."
    fi
else
    echo "ℹ️  .env file already exists, skipping creation."
fi

echo "✨ Setup complete! Your environment is ready."
echo "🐍 Virtual environment is activated at .venv"
echo "📝 Don't forget to check your .env file for correct configuration."

# Add venv activation to bashrc for persistent terminal sessions
echo "" >> ~/.bashrc
echo "# Auto-activate virtual environment" >> ~/.bashrc
echo "if [ -f ${PWD}/.venv/bin/activate ]; then" >> ~/.bashrc
echo "    source ${PWD}/.venv/bin/activate" >> ~/.bashrc
echo "fi" >> ~/.bashrc

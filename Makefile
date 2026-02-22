SHELL := /bin/bash

.PHONY: all build install jupyter clean

all: build

# Instala as dependências e configura o ambiente
install: build

build:
	@echo "🔧 Configurando o ambiente virtual e instalando dependências..."
	@bash activate.sh

# Inicia o Jupyter Lab
jupyter:
	@echo "📓 Iniciando Jupyter Lab..."
	@source activate.sh && jupyter lab --IdentityProvider.token="" --ServerApp.password=""

# Limpa arquivos temporários e caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .qml-env
	rm -rf qml.egg-info 
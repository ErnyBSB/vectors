# vectors
Assessing vector embeddings techniques
# Espaço de Embeddings 3D

Visualização interativa de embeddings vetoriais em espaço 3D, com similaridade coseno em tempo real.

## Acesso

Abra diretamente no browser: `index.html`

Ou acesse via GitHub Pages após ativar em **Settings → Pages → Branch: main → / (root)**.

## Interação

| Ação | Efeito |
|---|---|
| Arrastar | Rotacionar a cena |
| Scroll / pinça | Zoom |
| Clicar num ponto | Ver similaridade coseno com os vizinhos |
| Sliders | Ajustar separação, ruído e zoom |
| Botões de toggle | Eixos, linhas de origem, linhas de similaridade, auto-rotação |

## Grupos semânticos

32 palavras distribuídas em 4 clusters: **animais**, **tecnologia**, **culinária** e **emoções**.

Cada palavra possui um vetor de 8 dimensões gerado deterministicamente (seed 77), reduzido para 3D para visualização. A similaridade exibida é calculada no espaço original de 8D.

## Stack

HTML + CSS + JavaScript vanilla — sem dependências, sem build.

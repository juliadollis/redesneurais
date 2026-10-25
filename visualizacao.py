# Funções que só mostram coisas no notebook: contas formatadas, desenhos da rede,
# animação e gráfico do treino. Nenhuma conta da rede é feita aqui.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, Markdown, display

CORES = ['#8080ff', '#ffab4d', '#40dd40']   # azul, laranja, verde
CINZA = '#e6e6e6'
DX = 4.2   # espaço entre as camadas no desenho

plt.rcParams['figure.dpi'] = 110
plt.rcParams['animation.embed_limit'] = 60


# ---------------------------------------------------------------- contas escritas

def _mostrar(linhas):
    display(Markdown('\n'.join(linhas)))


def _d(nome):
    return f'\\frac{{\\partial L}}{{\\partial {nome}}}'


def _nome_a(c, k):
    return f'x_{k+1}' if c == 0 else f'a^{{({c})}}_{k+1}'


def conta_forward(c, W, b, z, a):
    C = len(W)
    linhas = []
    for j in range(len(z[c])):
        soma = ' + '.join(f'{W[c][j, k]:g} \\cdot {a[c-1][k]:.4f}' for k in range(len(a[c - 1])))
        if c < C:
            saida = f'a^{{({c})}}_{j+1} = f({z[c][j]:.4f}) = {a[c][j]:.4f}'
        else:
            exps = ' + '.join(f'e^{{{v:.4f}}}' for v in z[c])
            saida = f's_{j+1} = \\frac{{e^{{{z[c][j]:.4f}}}}}{{{exps}}} = {a[c][j]:.4f}'
        linhas.append(f'- $z^{{({c})}}_{j+1} = {soma} + {b[c][j]:g} = {z[c][j]:.4f} \\quad\\Rightarrow\\quad {saida}$')
    _mostrar(linhas)


def conta_loss(s, y, L):
    _mostrar([f'$s_1 = {s[0]:.4f}$, $s_2 = {s[1]:.4f}$ (somam {s.sum():.4f}); '
              f'classe prevista: **{np.argmax(s) + 1}**, classe certa: **{np.argmax(y) + 1}**',
              '',
              f'$L = -(y_1 \\log s_1 + y_2 \\log s_2) = -({y[0]:g} \\cdot \\log {s[0]:.4f} + {y[1]:g} \\cdot \\log {s[1]:.4f}) = {L:.4f}$'])


def conta_passo1(s, y, dL_ds):
    # + 0.0 pra não aparecer -0.0000 quando y = 0
    _mostrar([f'- ${_d(f"s_{i+1}")} = -\\frac{{y_{i+1}}}{{s_{i+1}}} = -\\frac{{{y[i]:g}}}{{{s[i]:.4f}}} = {dL_ds[i] + 0.0:.4f}$'
              for i in range(len(s))])


def conta_passo2_softmax(J, dL_ds, dL_dz):
    n = len(dL_dz)
    linhas = ' \\\\ '.join(' & '.join(f'{J[j, i]:.4f}' for i in range(n)) for j in range(n))
    texto = [f'$$J = \\begin{{pmatrix}} {linhas} \\end{{pmatrix}}$$', '']
    for i in range(n):
        soma = ' + '.join(f'({dL_ds[j] + 0.0:.4f}) \\cdot ({J[j, i]:.4f})' for j in range(n))
        texto.append(f'- ${_d(f"z^{{(3)}}_{i+1}")} = \\sum_j {_d("s_j")} \\, J_{{j,{i+1}}} = {soma} = {dL_dz[i]:.4f}$')
    _mostrar(texto)


def conta_passo2(c, dL_da, derivada, dL_dz):
    _mostrar([f'- ${_d(f"z^{{({c})}}_{i+1}")} = {_d(_nome_a(c, i))} \\cdot f\'(z^{{({c})}}_{i+1})'
              f' = {dL_da[i]:.4f} \\cdot {derivada[i]:.4f} = {dL_dz[i]:.4f}$'
              for i in range(len(dL_dz))])


def conta_passo3(c, grad_b):
    _mostrar([f'- ${_d(f"b^{{({c})}}_{i+1}")} = {_d(f"z^{{({c})}}_{i+1}")} = {grad_b[i]:.4f}$'
              for i in range(len(grad_b))])


def conta_passo4(c, dL_dz, a_ant, grad_W):
    _mostrar([f'- ${_d(f"w^{{({c})}}_{{{i+1},{j+1}}}")} = {_d(f"z^{{({c})}}_{i+1}")} \\cdot {_nome_a(c - 1, j)}'
              f' = {dL_dz[i]:.4f} \\cdot {a_ant[j]:.4f} = {grad_W[i, j]:.4f}$'
              for i in range(len(dL_dz)) for j in range(len(a_ant))])


def conta_passo5(c, dL_dz, W_c, dL_da_ant):
    linhas = []
    for i in range(len(dL_da_ant)):
        soma = ' + '.join(f'{dL_dz[j]:.4f} \\cdot {W_c[j, i]:g}' for j in range(len(dL_dz)))
        linhas.append(f'- ${_d(_nome_a(c - 1, i))} = {soma} = {dL_da_ant[i]:.4f}$')
    _mostrar(linhas)


# ---------------------------------------------------------------- desenho da rede

def _tamanhos(W):
    # neurônios por camada, com a entrada na posição 0
    return [W[1].shape[1]] + [W[c].shape[0] for c in sorted(W)]


def _nomes_camadas(C):
    return ['Entrada'] + [f'Camada {c} (hidden)' for c in range(1, C)] + [f'Camada {C} (saída)']


def _posicao(tam, c, i):
    return DX * c, ((tam[c] - 1) / 2 - i) * 2.4


def _posicao_rotulo(x0, y0, x1, y1):
    # põe o rótulo longe dos cruzamentos das linhas
    dx, dy = x1 - x0, y1 - y0
    distancia = (1.35 if abs(dy) < 2.5 else 1.9) / np.hypot(dx, dy)
    t = 0.5 if abs(dy) < 0.1 else distancia if dy < 0 else 1 - distancia
    return x0 + t * dx, y0 + t * dy


def _desenhar(tam, texto_no, texto_aresta, texto_bias, titulo, ax=None, apagadas=()):
    # texto_no[c][i]        -> escrito dentro do neurônio i da camada c
    # texto_aresta[c][j][k] -> escrito na conexão que sai do k e entra no j da camada c
    # texto_bias[c][j]      -> escrito embaixo do neurônio
    # apagadas              -> camadas que ficam cinza (usado na animação)
    C = len(tam) - 1
    mostrar = ax is None
    if mostrar:
        fig, ax = plt.subplots(figsize=(14, 8.5))
    cor = lambda c, i: CINZA if c in apagadas else CORES[i % len(CORES)]

    for c in range(1, C + 1):
        for j in range(tam[c]):
            for k in range(tam[c - 1]):
                (x0, y0), (x1, y1) = _posicao(tam, c - 1, k), _posicao(tam, c, j)
                ax.plot([x0, x1], [y0, y1], color=cor(c, j), lw=1.6, zorder=1)
                if texto_aresta[c][j][k]:
                    angulo = np.degrees(np.arctan2(y1 - y0, x1 - x0))
                    ax.text(*_posicao_rotulo(x0, y0, x1, y1), texto_aresta[c][j][k],
                            rotation=angulo, rotation_mode='anchor', ha='center', va='center',
                            fontsize=9, color='black', zorder=3,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cor(c, j), lw=1))

    for c in range(C + 1):
        for i in range(tam[c]):
            xc, yc = _posicao(tam, c, i)
            ax.add_patch(plt.Circle((xc, yc), 0.62, fc=cor(c, i), ec='black', lw=1.5, zorder=2))
            ax.text(xc, yc, texto_no[c][i], ha='center', va='center', fontsize=10, zorder=4)
            if c > 0 and texto_bias[c][i]:
                ax.text(xc, yc - 0.8, texto_bias[c][i], ha='center', va='top', fontsize=9.5,
                        color=cor(c, i), fontweight='bold', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        ax.text(DX * c, 4.2, _nomes_camadas(C)[c], ha='center', fontsize=12)

    ax.set_title(titulo, fontsize=14, pad=12)
    ax.set_aspect('equal')
    ax.set_xlim(-1, DX * C + 1.2)
    ax.set_ylim(-4, 4.8)
    ax.axis('off')
    if mostrar:
        plt.show()


def _rotular(valores, fmt):
    # {c: array} -> {c: lista de textos}
    textos = {}
    for c, arr in valores.items():
        if arr.ndim == 2:
            textos[c] = [[fmt(c, j, k, v) for k, v in enumerate(linha)] for j, linha in enumerate(arr)]
        else:
            textos[c] = [fmt(c, j, None, v) for j, v in enumerate(arr)]
    return textos


def _txt_entrada(x):
    return [f'$x_{i+1}$\n{v:g}' for i, v in enumerate(x)]


def _textos_forward(x, W, b, z, a):
    C = len(W)
    no = {0: _txt_entrada(x),
          **{c: [f'z={z[c][i]:.3f}\n{"s" if c == C else "a"}={a[c][i]:.3f}' for i in range(len(z[c]))] for c in z}}
    return no, _rotular(W, lambda c, j, k, v: f'{v:g}'), _rotular(b, lambda c, j, k, v: f'b={v:g}')


def _textos_backward(x, dL_dz, grad_W, grad_b):
    no = {0: _txt_entrada(x), **{c: [f'∂L/∂z\n{v:+.4f}' for v in dL_dz[c]] for c in dL_dz}}
    return (no, _rotular(grad_W, lambda c, j, k, v: f'{v:+.4f}'),
            _rotular(grad_b, lambda c, j, k, v: f'∂L/∂b={v:+.4f}'))


def desenhar_parametros(x, W, b):
    C = len(W)
    no = {0: _txt_entrada(x),
          **{c: [f'$z^{{({c})}}_{i+1}\\,|\\,a^{{({c})}}_{i+1}$' for i in range(len(b[c]))] for c in range(1, C)},
          C: [f'$z^{{({C})}}_{i+1}\\,|\\,s_{i+1}$' for i in range(len(b[C]))]}
    _desenhar(_tamanhos(W), no,
              _rotular(W, lambda c, j, k, v: f'$w^{{({c})}}_{{{j+1},{k+1}}}$={v:g}'),
              _rotular(b, lambda c, j, k, v: f'$b^{{({c})}}_{j+1}$={v:g}'),
              'Pesos e biases iniciais')


def desenhar_forward(x, W, b, z, a, L):
    _desenhar(_tamanhos(W), *_textos_forward(x, W, b, z, a),
              f'Forward pass: z e a em cada neurônio, s na saída   (L = {L:.4f})')


def desenhar_backward(x, W, dL_dz, grad_W, grad_b):
    _desenhar(_tamanhos(W), *_textos_backward(x, dL_dz, grad_W, grad_b),
              'Backward pass: ∂L/∂z nos neurônios, ∂L/∂w nas conexões, ∂L/∂b embaixo')


# ---------------------------------------------------------------- animação

def _quadros(C):
    # cada quadro é (fase, camada, t)
    # t vai de 0 a 1 enquanto as bolinhas andam pelas conexões; None é quadro parado
    movimento = list(np.linspace(0, 1, 8))
    pausa = [None] * 3

    quadros = [('forward', 0, None)] * 2                        # só a entrada acesa
    for c in range(1, C + 1):
        quadros += [('forward', c, t) for t in movimento + pausa]  # bolinhas indo pra camada c
    quadros += [('forward', C + 1, None)] * 5                   # mostra a loss
    for c in range(C, 0, -1):
        quadros += [('backward', c, None)] * 5                  # gradientes da camada c aparecem
        if c > 1:
            quadros += [('backward', c, t) for t in movimento]  # bolinhas voltando pra camada c-1
    quadros += [('backward', 1, None)] * 5
    return quadros


def _estado_da_camada(c, fase, c_atual, t):
    if c == 0:
        return 'forward'
    if fase == 'forward':
        # já calculada, ou acabou de ser calculada (bolinhas chegaram)
        return 'forward' if c < c_atual or (c == c_atual and t is None) else 'vazia'
    # no backward as camadas da direita já têm gradiente
    return 'backward' if c >= c_atual else 'forward'


def _titulo_do_quadro(fase, c_atual, t, C, L):
    if fase == 'forward':
        if c_atual == 0:
            return 'Forward pass: entrada'
        if c_atual > C:
            return f'Fim do forward: L = {L:.4f}'
        return f'Forward pass: calculando a camada {c_atual}'
    if t is not None:
        return f'Backpropagation: levando ∂L/∂a para a camada {c_atual - 1}'
    return f'Backpropagation: gradientes da camada {c_atual}'


def animar(x, W, b, z, a, L, dL_dz, grad_W, grad_b):
    tam = _tamanhos(W)
    C = len(W)
    vazio = ({c: [''] * tam[c] for c in range(C + 1)},
             {c: [[''] * tam[c - 1] for _ in range(tam[c])] for c in range(1, C + 1)},
             {c: [''] * tam[c] for c in range(1, C + 1)})
    textos = {'vazia': vazio,
              'forward': _textos_forward(x, W, b, z, a),
              'backward': _textos_backward(x, dL_dz, grad_W, grad_b)}
    quadros = _quadros(C)

    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=60)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)   # tira a margem branca

    def desenhar_quadro(n):
        fase, c_atual, t = quadros[n]
        ax.clear()
        estados = {c: _estado_da_camada(c, fase, c_atual, t) for c in range(C + 1)}
        _desenhar(tam,
                  texto_no={c: textos[estados[c]][0][c] for c in range(C + 1)},
                  texto_aresta={c: textos[estados[c]][1][c] for c in range(1, C + 1)},
                  texto_bias={c: textos[estados[c]][2][c] for c in range(1, C + 1)},
                  titulo=_titulo_do_quadro(fase, c_atual, t, C, L),
                  ax=ax, apagadas=[c for c in estados if estados[c] == 'vazia'])

        # bolinhas: pretas indo pra frente, vermelhas voltando
        if t is not None:
            origem, destino = (c_atual - 1, c_atual) if fase == 'forward' else (c_atual, c_atual - 1)
            for i in range(tam[origem]):
                for j in range(tam[destino]):
                    (x0, y0), (x1, y1) = _posicao(tam, origem, i), _posicao(tam, destino, j)
                    ax.plot(x0 + t * (x1 - x0), y0 + t * (y1 - y0), 'o', ms=10, mec='black', zorder=5,
                            color='#333333' if fase == 'forward' else '#e03030')

    anim = animation.FuncAnimation(fig, desenhar_quadro, frames=len(quadros), interval=110)
    plt.close(fig)   # senão aparece uma figura parada a mais
    return HTML(anim.to_jshtml(default_mode='once'))


# ---------------------------------------------------------------- treino

def grafico_treino(historico_L, historico_s):
    historico_s = np.array(historico_s)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(historico_L, color=CORES[0], lw=2)
    ax1.set(title='Loss ao longo do treino', xlabel='época', ylabel='L')

    ax2.plot(historico_s[:, 0], color=CORES[0], lw=2, label='$s_1$ (classe certa)')
    ax2.plot(historico_s[:, 1], color=CORES[1], lw=2, label='$s_2$')
    ax2.axhline(1, color=CORES[0], ls='--', lw=1)
    ax2.axhline(0, color=CORES[1], ls='--', lw=1)
    ax2.set(title='Probabilidade de cada classe ao longo do treino', xlabel='época')
    ax2.legend()

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    plt.show()

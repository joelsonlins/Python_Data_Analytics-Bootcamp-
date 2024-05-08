capacidade_atual, aumento_percentual = map(int, input().split())

# // TODO: Calcule a nova capacidade do disco de Mithril

aumento_teraflops = capacidade_atual * aumento_percentual / 100

nova_capacidade = capacidade_atual + aumento_teraflops

# // TODO: Imprima a nova capacidade



print(int(nova_capacidade))
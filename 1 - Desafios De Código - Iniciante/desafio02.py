# Lista para armazenar os itens
itens = []

#//TODO: Solicite os itens ao usuário
itens.append(input())
itens.append(input())
itens.append(input())

# Exibe a lista de itens
print("Lista de itens:")
for item in itens:
    print(f"- {item}")
def AQ11(E1, E2, G):
    if len(E1) == 0 or all([is_covered(ex, G) for ex in E1]):
        return G
    else:
        ex = E1[0]
        for c in E2:
            if is_consistent([ex], c):
                new_clause = merge([ex], c)
                E1_new = [ex for ex in E1 if not is_covered(ex, [new_clause])]
                E2_new = [c for c in E2 if not is_more_general([new_clause], c)]
                G_new = G + [new_clause]
                result = AQ11(E1_new, E2_new, G_new)
                if result is not None:
                    return result
        return None


def is_consistent(c1, c2):
    for literal in c1 + c2:
        if negate_literal(literal) in c1 + c2:
            return False
    return True


def negate_literal(literal):
    if literal.startswith('~'):
        return literal[1:]
    else:
        return '~' + literal


def merge(c1, c2):
    literals = set(c1 + c2)
    for literal in literals:
        if negate_literal(literal) in literals:
            return None
    return list(literals)


def is_more_general(c1, c2):
    if len(c1) != 1 or len(c2) != 1:
        return False
    for i in range(len(c1[0])):
        if c1[0][i] != c2[0][i] and c1[0][i] != '?':
            return False
    return True


def is_covered(ex, G):
    for clause in G:
        if all([ex[i] == clause[i] for i in range(len(ex))]):
            return True
    return False


dataset = [    ['A', '~B', 'C'],
    ['~A', 'B', 'C'],
    ['~A', '~B', 'D'],
    ['A', 'B', '~C'],
    ['~A', '~B', '~C'],
    ['A', '~C', 'D'],
]

G = AQ11(dataset, [], [])

print(G)
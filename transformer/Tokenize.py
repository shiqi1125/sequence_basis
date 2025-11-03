import json

base = ['A', 'C', 'G', 'T', 'N']
base_to_idx = {'A':0, 'C':1, 'G':2, 'T':3, 'N':0}

def permutation(k):
    result = []
    def backtrack(k_mers):
        if len(k_mers) == k:
            result.append(k_mers.copy())
            return
        for elem in base:
            k_mers.append(elem)
            backtrack(k_mers)
            k_mers.pop()
    backtrack([])
    return result

def tokenization_by_k(k):
    permutations = permutation(k)
    token_ids = {}
    for elem in permutations:
        k_mers = ''
        idx = 0
        count = k - 1
        for mer in elem:
            k_mers = k_mers + mer
            idx = idx + 4**(count) * base_to_idx.get(mer, 0)
            count = count - 1
        token_ids[k_mers] = idx
    return token_ids
    
def generate_token_ids(k):
    data = tokenization_by_k(k)
    with open('6-mers.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    generate_token_ids(6)
    
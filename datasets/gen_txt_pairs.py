"""
Generate data pairs text file
"""


def main():
    file_name = 'CUFED5_pairs_multi.txt'
    num_input = 126
    num_ref = 5

    with open(file_name, 'w') as f:
        for input_idx in range(num_input):
            pair_group = []
            for ref_idx in range(1, 1 + num_ref):
                pair = f'{input_idx:03d}_0.png {input_idx:03d}_{ref_idx}.png\n'
                pair_group.append(pair)
            f.writelines(pair_group)


if __name__ == '__main__':
    main()
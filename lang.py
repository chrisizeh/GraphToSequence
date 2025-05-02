import numpy as np

# Tokens for model to know if sequence is start or and of text
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, num_nodes):
        self.word2index = {"(": 3, ")": 4, "*": 5, ".": 6}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "(", 4: ")", 5: "*", 6: "."}
        self.n_words = 7  # Count SOS and EOS

        for i in range(num_nodes):
            self.index2word[self.n_words] = str(i)
            self.word2index[str(i)] = self.n_words
            self.n_words += 1

    # transform a string into an encoded numpy array.
    # If max_len provided, string will be truncated
    # If min_len provided, padding is added to array
    def seq2arr(self, seq, max_len=-1, min_len=0):
        if max_len <= 0:
            max_len = len(seq)
        elif (max_len > len(seq)):
            max_len = len(seq)

        arr = np.zeros(max(max_len, min_len), dtype=np.int64)
        for j in range(max_len):
            arr[j] = self.word2index[seq[j]]

        return arr

    def seq2arrWithStart(self, seq, max_len=-1, min_len=0):
        arr = np.zeros(max(max_len, min_len) + 1, dtype=np.int64)
        arr[0] = 1
        arr[1:] = self.seq2arr(seq, max_len, min_len)
        return arr

    # WIP

    def subseq2arr(self, seq, index, length=-1, max_len=-1, min_len=0):
        arr = np.zeros(max(max_len, min_len) + 2, dtype=np.int64)

        if (index == 0):
            arr[0] = 1

        arr[1:] = self.seq2arr(seq[index:index+length], max_len, min_len)

        if (index+length > len(seq)):
            arr[len(seq)] = 2
            return arr
        else:
            return arr[:-1]


if __name__ == "__main__":
    lang = Lang(14)

    print(lang.seq2arr(".11.(12.6."))
    print(lang.seq2arr(".11.(12.6.", max_len=4))
    print(lang.seq2arr(".", min_len=3))
    print(lang.seq2arr("", min_len=3))
    print(lang.seq2arrWithStart("", min_len=3))

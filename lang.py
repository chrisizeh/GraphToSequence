import numpy as np

# Tokens for model to know if sequence is start or and of text
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, num_nodes):
        self.word2index = {";": 3, "(": 4, ")": 5, "*": 6, ".": 7}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: ";", 4: "(", 5: ")", 6: "*", 7: "."}
        self.n_words = 8  # Count SOS, EOS and PAD

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

    def subseq2arr(self, seq, arr_length, index, length):
        arr = np.zeros(arr_length, dtype=np.int64)
        stop = 0

        if (index+length >= len(seq)):
            arr[-1] = 2
            stop = 1
            length = len(seq) - index

        if (length + stop > arr_length):
            length = arr_length - stop

        if (index < 0):
            return arr
        elif (index == 0):
            arr[arr_length-length-stop-1] = 1

        if length > 0:
            arr[arr_length-length-stop:arr_length-stop] = self.seq2arr(seq[index:index+length])

        return arr

    def arr2seq(self, arr, ignoreTokens=False):
        if (ignoreTokens):
            return "".join([self.index2word[idx.item()] for idx in arr if idx > 2])
        return "".join([self.index2word[idx.item()] for idx in arr])


if __name__ == "__main__":
    lang = Lang(14)

    print(lang.seq2arr(".11.(12.6."))
    print(lang.seq2arr(".11.(12.6.", max_len=4))
    print(lang.seq2arr(".", min_len=3))
    print(lang.seq2arr("", min_len=3))
    print(lang.subseq2arr("", 4, 0, 0))
    print(lang.subseq2arr("1234", 4, 0, 1))
    print(lang.subseq2arr(".11.(12.6.", 5, 0, 4))
    print(lang.subseq2arr(".11.(", 7, 0, 5))
    print(lang.subseq2arr("4.8.;13.11.(12.6.(7.9.10.5.*6.3.1.2.*11.))", 11, 31, 11))

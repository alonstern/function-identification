import glob
import os

import numpy
import tqdm
from elftools.elf.elffile import ELFFile
from torch.utils import data

FILE_START = 256
FILE_END = 257


class FunctionIdentificationDataset(data.Dataset):
    def __init__(self, root_directory, block_size, padding_size):
        self._padding_size = padding_size
        data, labels = self._preprocess_data(root_directory)
        self._data_blocks, self._labels_blocks = self._split_to_blocks(data, labels, block_size)

    def __len__(self):
        return len(self._data_blocks)

    def __getitem__(self, idx):
        return self._data_blocks[idx], self._labels_blocks[idx]

    def _preprocess_data(self, root_directory):
        files_data = []
        files_labels = []
        binaries_path = glob.glob(os.path.join(root_directory, "*", "binary", "*"))
        for binary_path in tqdm.tqdm(binaries_path):
            with open(binary_path, "rb") as binary_file:
                binary_elf = ELFFile(binary_file)
                data = numpy.array(list(binary_elf.get_section_by_name(".text").data()), dtype=int)
                labels = self._generate_lables(binary_elf)
                files_data.append(data)
                files_labels.append(labels)
        return files_data, files_labels

    def _generate_lables(self, binary_elf):
        text_section = binary_elf.get_section_by_name(".text")
        labels = numpy.zeros(text_section.data_size, dtype=int)

        symbol_starts = [symbol[0] - text_section["sh_addr"] for symbol in self._get_function_addresses(binary_elf)]
        labels[symbol_starts] = 1
        return labels

    @staticmethod
    def _get_function_addresses(binary_elf):
        symbol_table = binary_elf.get_section_by_name(".symtab")
        return [(symbol["st_value"], symbol["st_size"]) for symbol in symbol_table.iter_symbols() if
                symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_size"] != 0]

    def _split_to_blocks(self, data, labels, block_size):
        data_blocks = []
        labels_blocks = []
        for file_data, file_labels in zip(data, labels):
            for start_index in range(0, len(file_data), block_size):
                data_blocks.append(self._get_padded_data(file_data, start_index, block_size))
                labels_blocks.append(file_labels[start_index: start_index + block_size])

        return data_blocks, labels_blocks

    def _get_padded_data(self, file_data, index, block_size):
        left_padding_number = int(self._padding_size / 2)
        right_padding_number = self._padding_size - left_padding_number

        left_padding = numpy.array([FILE_START] * (left_padding_number - index), dtype=int)
        right_padding = numpy.array([FILE_END] * (right_padding_number - max(file_data.size - index - block_size, 0)), dtype=int)
        block = file_data[max(index - left_padding_number, 0): index + block_size + right_padding_number]

        return numpy.concatenate([left_padding, block, right_padding])


if __name__ == '__main__':
    dataset = FunctionIdentificationDataset("/home/alon/function-identification-dataset/security.ece.cmu.edu/byteweight/elf_32", 1000, 5)

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
        data, tags = self._preprocess_data(root_directory)
        self._data_blocks, self._tags_blocks = self._split_to_blocks(data, tags, block_size, padding_size)

    def __len__(self):
        return len(self._data_blocks)

    def __getitem__(self, idx):
        return self._data_blocks[idx], self._tags_blocks[idx]

    def _preprocess_data(self, root_directory):
        files_data = []
        files_tags = []
        # Iterates over every binary in the dataset
        for binary_path in tqdm.tqdm(glob.glob(os.path.join(root_directory, "*", "binary", "*"))):
            with open(binary_path, "rb") as binary_file:
                binary_elf = ELFFile(binary_file)

                # Extract the code from the binary.
                data = self._generate_data(binary_elf)

                # Extract the tags of each byte in the binary code (1 if it is a start of a function, 0 otherwise).
                tags = self._generate_tags(binary_elf)

                files_data.append(data)
                files_tags.append(tags)
        return files_data, files_tags

    def _generate_data(self, binary_elf: ELFFile):
        return numpy.array(list(binary_elf.get_section_by_name(".text").data()), dtype=int)

    def _generate_tags(self, binary_elf: ELFFile):
        text_section = binary_elf.get_section_by_name(".text")

        # text_section["sh_addr"] is the address of the .text section.
        # We need the addresses of the symbols to be relative to the .text section so we subtract sh_addr from them.
        function_addresses = [function_address - text_section["sh_addr"] for function_address in
                              self._get_function_addresses(binary_elf)]

        tags = numpy.zeros(text_section.data_size, dtype=int)
        tags[function_addresses] = 1
        return tags

    @staticmethod
    def _get_function_addresses(binary_elf):
        symbol_table = binary_elf.get_section_by_name(".symtab")

        # st_value is the address of the symbol in the binary.
        # There are more types of symbol than function so we make sure we only get the function symbols
        return [symbol["st_value"] for symbol in symbol_table.iter_symbols()
                if symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_size"] != 0]

    def _split_to_blocks(self, data, tags, block_size, padding_size):
        data_blocks = []
        tags_blocks = []
        for file_data, file_tags in zip(data, tags):
            for start_index in range(0, len(file_data), block_size):
                data_blocks.append(self._get_padded_data(file_data, start_index, block_size, padding_size))
                tags_blocks.append(file_tags[start_index: start_index + block_size])

        return data_blocks, tags_blocks

    def _get_padded_data(self, file_data, index, block_size, padding_size):
        left_padding_number = int(padding_size / 2)
        right_padding_number = padding_size - left_padding_number

        # If there is data available before the block we will use it for padding. Otherwise we will use FILE_START.
        # Same for FILE_END.
        left_padding = numpy.array([FILE_START] * (left_padding_number - index), dtype=int)
        right_padding = numpy.array([FILE_END] * (right_padding_number - max(file_data.size - index - block_size, 0)), dtype=int)
        block = file_data[max(index - left_padding_number, 0): index + block_size + right_padding_number]

        return numpy.concatenate([left_padding, block, right_padding])

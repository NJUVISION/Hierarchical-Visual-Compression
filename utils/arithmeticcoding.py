
import numpy as np
import sys
python3 = sys.version_info.major >= 3
import time

class ArithmeticCoderBase(object):
	def __init__(self, statesize):
		self.STATE_SIZE = statesize
		# Maximum range (high+1-low) during coding (trivial), which is 2^STATE_SIZE = 1000...000.
		self.MAX_RANGE = 1 << self.STATE_SIZE
		# Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
		self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
		# Maximum allowed total from a frequency table at all times during coding. This differs from Java
		# and C++ because Python's native bigint avoids constraining the size of intermediate computations.
		self.MAX_TOTAL = self.MIN_RANGE
		# Bit mask of STATE_SIZE ones, which is 0111...111.
		self.MASK = self.MAX_RANGE - 1
		# The top bit at width STATE_SIZE, which is 0100...000.
		self.TOP_MASK = self.MAX_RANGE >> 1
		# The second highest bit at width STATE_SIZE, which is 0010...000. This is zero when STATE_SIZE=1.
		self.SECOND_MASK = self.TOP_MASK >> 1
		
		# -- State fields --
		# Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
		self.low = 0
		# High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
		self.high = self.MASK
	

	def update(self,  cumul, symbol):
		low = self.low
		high = self.high
		range = high - low + 1
		total = cumul[-1].item()
		symlow = cumul[symbol].item()
		symhigh = cumul[symbol+1].item()
		newlow  = low + symlow  * range // total
		newhigh = low + symhigh * range // total - 1
		self.low = newlow
		self.high = newhigh
		while ((self.low ^ self.high) & self.TOP_MASK) == 0:
			self.shift()
			self.low = (self.low << 1) & self.MASK
			self.high = ((self.high << 1) & self.MASK) | 1
		
		while (self.low & ~self.high & self.SECOND_MASK) != 0:
			self.underflow()
			self.low = (self.low << 1) & (self.MASK >> 1)
			self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1
	
	def shift(self):
		raise NotImplementedError()
	
	

	def underflow(self):
		raise NotImplementedError()



# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):
	
	# Constructs an arithmetic coding encoder based on the given bit output stream.
	def __init__(self, statesize, bitout):
		super(ArithmeticEncoder, self).__init__(statesize)
		# The underlying bit output stream.
		self.output = bitout
		# Number of saved underflow bits. This value can grow without bound.
		self.num_underflow = 0
	
	
	# Encodes the given symbol based on the given frequency table.
	# This updates this arithmetic coder's state and may write out some bits.
	def write(self, cumul, symbol):

                self.update(cumul, symbol)

	
	

	def finish(self):
		self.output.write(1)

	
	
	def shift(self):
		bit = self.low >> (self.STATE_SIZE - 1)
		self.output.write(bit)
		for _ in range(self.num_underflow):
			self.output.write(bit ^ 1)
		self.num_underflow = 0

	
	def underflow(self):
		self.num_underflow += 1



# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):
	
	def __init__(self, statesize, bitin):
		super(ArithmeticDecoder, self).__init__(statesize)
		# The underlying bit input stream.
		self.input = bitin
		# The current raw code bits being buffered, which is always in the range [low, high].
		self.code = 0
		for _ in range(self.STATE_SIZE):
			self.code = self.code << 1 | self.read_code_bit()
	
	

	def read(self, cumul, alphabet_size):
		total = cumul[-1].item()
		range = self.high - self.low + 1
		offset = self.code - self.low
		value = ((offset + 1) * total - 1) // range
		start = 0
		end = alphabet_size
		while end - start > 1:
			middle = (start + end) >> 1
			if cumul[middle] > value:
				end = middle
			else:
				start = middle
		
		symbol = start
		self.update(cumul, symbol)
		return symbol
	
	
	def shift(self):
		self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()
	
	
	def underflow(self):
		self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()
	
	
	def read_code_bit(self):
		temp = self.input.read()
		if temp == -1:
			temp = 0
		return temp


class BitCollector:
    def __init__(self):
        self.bits = []

    def write(self, bit):
        self.bits.append(bit)

    def get_bits(self):
        return self.bits
    
class BitListReader:
	def __init__(self, bits_list: list):
		self.bits = bits_list
		self.index = 0  

	def read(self):
		if self.index >= len(self.bits):
			return -1 
		
		bit = self.bits[self.index]
		self.index += 1
		return bit

	def get_consumed_bits_count(self):
		return self.index

class BitOutputStream:
    def __init__(self, file):
        self.file = file
        self.buffer = 0
        self.bit_count = 0

    def write(self, bit):
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.buffer = (self.buffer << 1) | bit
        self.bit_count += 1
        if self.bit_count == 8:
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.bit_count = 0

    def flush(self):
        if self.bit_count > 0:
            self.buffer <<= (8 - self.bit_count)
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.bit_count = 0

    def close(self):
        self.flush()
        self.file.close()

class BitInputStream:
    def __init__(self, file):
        self.file = file
        self.buffer = 0
        self.bit_count = 0

    def read(self):
        if self.bit_count == 0:
            byte_data = self.file.read(1)
            if not byte_data:
                return -1  # End of stream
            self.buffer = byte_data[0]
            self.bit_count = 8
        
        # Read the most significant bit from the buffer
        bit = (self.buffer >> (self.bit_count - 1)) & 1
        self.bit_count -= 1
        return bit
        
    def close(self):
        self.file.close()

def probabilities_to_cumul(p0, total_freq=2**16):
    p0 = max(0.0, min(1.0, p0))
    
    freq_0 = max(1, int(round(p0 * total_freq)))
    freq_1 = max(1, int(round((1.0 - p0) * total_freq)))
    current_total = freq_0 + freq_1
    cumul = np.array([0, freq_0, current_total], dtype=np.uint64)
    return cumul

def compress_to_bit_list(bit_source, probabilities_of_zero):

    bit_collector = BitCollector()
    encoder = ArithmeticEncoder(statesize=32, bitout=bit_collector)

    for i, bit in enumerate(bit_source):
        p0 = probabilities_of_zero[i]
        cumul = probabilities_to_cumul(p0)
        encoder.write(cumul, bit)

    encoder.finish()
    
    return bit_collector.get_bits()


def decompress_from_bit_list(exact_bit_list, num_bits, probabilities_of_zero):
    bit_reader = BitListReader(exact_bit_list)
    decoder = ArithmeticDecoder(statesize=32, bitin=bit_reader)
    
    decoded_bits = []
    alphabet_size = 2
    
    for i in range(num_bits):
        p0 = probabilities_of_zero[i]
        cumul = probabilities_to_cumul(p0)
        
        bit = decoder.read(cumul, alphabet_size)
        decoded_bits.append(bit)
        
    return decoded_bits
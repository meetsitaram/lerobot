# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def encode_sign_magnitude(value: int, sign_bit_index: int):
    """
    https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude
    """
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(f"Magnitude {magnitude} exceeds {max_magnitude} (max for {sign_bit_index=})")

    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int):
    """
    https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude
    """
    direction_bit = (encoded_value >> sign_bit_index) & 1
    magnitude_mask = (1 << sign_bit_index) - 1
    magnitude = encoded_value & magnitude_mask
    return -magnitude if direction_bit else magnitude


def encode_twos_complement(value: int, n_bytes: int):
    """
    https://en.wikipedia.org/wiki/Signed_number_representations#Two%27s_complement
    """

    bit_width = n_bytes * 8
    min_val = -(1 << (bit_width - 1))
    max_val = (1 << (bit_width - 1)) - 1

    # Fast path: value already fits in signed range
    if min_val <= value <= max_val:
        if value >= 0:
            return value
        return (1 << bit_width) + value

    # Try a common normalization: maybe the caller passed an unsigned value or a
    # value that wrapped incorrectly (for example -4294963418). Normalize by
    # masking to the lowest `bit_width` bits and check whether that corresponds
    # to a valid two's-complement signed value.
    mask = (1 << bit_width) - 1
    normalized = value & mask

    # Convert normalized back to signed candidate
    signed_candidate = normalized - (1 << bit_width) if (normalized & (1 << (bit_width - 1))) else normalized

    if min_val <= signed_candidate <= max_val:
        # Return the unsigned representation (lowest-bit_width bits) which is
        # the two's-complement encoding of the signed candidate.
        return normalized

    raise ValueError(
        f"Value {value} out of range for {n_bytes}-byte two's complement: [{min_val}, {max_val}]"
    )


def decode_twos_complement(value: int, n_bytes: int) -> int:
    """
    https://en.wikipedia.org/wiki/Signed_number_representations#Two%27s_complement
    """
    bits = n_bytes * 8
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        value -= 1 << bits
    return value

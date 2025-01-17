# BSD 3-Clause License
#
# Copyright (c) 2019-2022, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Contains base classes for various models.

Because implementation may involve parallelization, these classes should only contain pre-processing and post-processing.

"""

from __future__ import annotations

from romcomma.typing_ import *
import shutil
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path
from warnings import warn
from numpy import array, atleast_2d, broadcast_to, sqrt, einsum, exp, prod, eye, shape, transpose, zeros, delete, diag, diagonal, reshape, full, ones, empty, \
    arange, sum, concatenate, copyto, sign
from pandas import DataFrame, MultiIndex, concat
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve, qr
import json
from romcomma.data import Frame, Fold
from copy import deepcopy
from romcomma import distribution


class Parameters(ABC):
    """ Abstraction of the parameters of a Model. Essentially a NamedTuple backed by files in a folder.
    Note that file writing is lazy, it must be called explicitly, but the Parameters are designed for chained calls."""

    @classmethod
    @property
    @abstractmethod
    def Values(cls) -> Type[NamedTuple]:
        """ The NamedTuple underpinning this Parameters set."""

    @classmethod
    def make(cls, iterable: Iterable) -> Parameters:
        """ Wrapper for namedtuple._make. See https://docs.python.org/3/library/collections.html#collections.namedtuple."""
        return cls.Values._make(iterable)

    @classmethod
    @property
    def fields(cls) -> Tuple[str, ...]:
        """ Wrapper for namedtuple._fields. See https://docs.python.org/3/library/collections.html#collections.namedtuple."""
        return cls.Values()._fields

    @classmethod
    @property
    def field_defaults(cls) -> Dict[str, Any]:
        """ Wrapper for namedtuple._field_defaults. See https://docs.python.org/3/library/collections.html#collections.namedtuple."""
        return cls.Values()._field_defaults

    @property
    def folder(self) -> Path:
        """ The folder containing the Parameters. """
        return self._folder

    @property
    def values(self) -> Values:
        """ Gets or Sets the NamedTuple of the Parameters."""
        return self._values

    @values.setter
    def values(self, value: Values):
        """ Gets or Sets the NamedTuple of the Parameters."""
        self._values = self.Values(*(atleast_2d(val) for val in value))

    def as_dict(self) -> Dict[str, Any]:
        """ Wrapper for namedtuple._asdict. See https://docs.python.org/3/library/collections.html#collections.namedtuple."""
        return self._values._asdict()

    def replace(self, **kwargs: NP.Matrix) -> Parameters:
        """ Replace selected field values in this Parameters. Does not write to folder.

        Args:
            **kwargs: key=value pairs of NamedTuple fields, precisely as in NamedTuple._replace(**kwargs).
        Returns: ``self``, for chaining calls.
        """
        for key, value in kwargs.items():
            kwargs[key] = atleast_2d(value)
        self._values = self._values._replace(**kwargs)
        return self

    def broadcast_value(self, model_name: str, field: str, target_shape: Tuple[int, int], is_diagonal: bool = True,
                        folder: Optional[PathLike] = None) -> Parameters:
        """ Broadcast a parameter value.

        Args:
            model_name: Used only in error reporting.
            field: The name of the field whose value we are broadcasting.
            target_shape: The shape to broadcast to.
            is_diagonal: Whether to zero the off-diagonal elements of a square matrix.
            folder:

        Returns: Self, for chaining calls.
        Raises:
            IndexError: If broadcasting is impossible.
        """
        replacement = {field: getattr(self.values, field)}
        try:
            replacement[field] = array(broadcast_to(replacement[field], target_shape))
        except ValueError:
            raise IndexError(f'The {model_name} {field} has shape {replacement[field].shape} '
                             f' which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            replacement[field] = diag(diagonal(replacement[field]))
        return self.replace(**replacement).write(folder)

    def _set_folder(self, folder: Optional[PathLike] = None):
        """ Set the file location for these Parameters.

        Args:
            folder: The file location is changed to ``folder`` unless ``folder`` is ``None`` (the default).
        """
        if folder is not None:
            self._folder = Path(folder)
            self._csv = tuple((self._folder / field).with_suffix(".csv") for field in self.fields)

    def read(self) -> Parameters:
        """ Read Parameters from their csv files.

        Returns: ``self``, for chaining calls.
        Raises:
            AssertionError: If self._csv is not set.
        """
        assert getattr(self, '_csv', None) is not None, 'Cannot perform file operations before self._folder and self._csv are set.'
        self._values = self.Values(**{key: Frame(self._csv[i], header=[0]).df.values for i, key in enumerate(self.fields)})
        return self

    def write(self, folder: Optional[PathLike] = None) -> Parameters:
        """  Write Parameters to their csv files.

        Args:
            folder: The file location is changed to ``folder`` unless ``folder`` is ``None`` (the default).
        Returns: ``self``, for chaining calls.
        Raises:
            AssertionError: If self._csv is not set.
        """
        self._set_folder(folder)
        assert getattr(self, '_csv', None) is not None, 'Cannot perform file operations before self._folder and self._csv are set.'
        dummy = tuple(Frame(self._csv[i], DataFrame(p)) for i, p in enumerate(self._values))
        return self

    def __init__(self, folder: Optional[PathLike] = None, **kwargs: NP.Matrix):
        """ Parameters Constructor. Shouldn't need to be overridden. Does not write to file.

        Args:
            folder: The folder to store the parameters.
            **kwargs: key=value initial pairs of NamedTuple fields, precisely as in NamedTuple(**kwargs). It is the caller's responsibility to ensure
                that every value is of type NP.Matrix. Missing fields receive their defaults, so Parameters(folder) is the default parameter set.
        """
        for key, value in kwargs.items():
            kwargs[key] = atleast_2d(value)
        self._set_folder(folder)
        self._values = self.Values(**kwargs)


class Model(ABC):
    """ Abstract base class for any model. This base class implements generic file storage and parameter handling.
    The latter is dealt with by each subclass overriding ``Parameters.Values`` with its own ``Type[NamedTuple]``
    defining the parameter set it takes.``model.parameters.values`` is a ``Values=Type[NamedTuple]`` of NP.Matrices.
    """

    @staticmethod
    def delete(folder: PathLike, ignore_errors: bool = True):
        """ Remove a folder tree, using shutil.

        Args:
            folder: Root of the tree to remove.
            ignore_errors: Boolean.
        """
        shutil.rmtree(folder, ignore_errors=ignore_errors)

    @staticmethod
    def copy(src_folder: PathLike, dst_folder: PathLike, ignore_errors: bool = True):
        """ Copy a folder tree, using shutil.

        Args:
            src_folder: Source root of the tree to copy.
            dst_folder: Destination root.
            ignore_errors: Boolean
        """
        shutil.rmtree(dst_folder, ignore_errors=ignore_errors)
        shutil.copytree(src=src_folder, dst=dst_folder)

    @classmethod
    @property
    @abstractmethod
    def DEFAULT_OPTIONS(cls) -> Dict[str, Any]:
        """ Default options."""

    @property
    def folder(self) -> Path:
        """ The model folder."""
        return self._folder

    @property
    def _options_json(self) -> Path:
        return self._folder / "options.json"

    @property
    def parameters(self) -> Parameters:
        """ Sets or gets the model parameters, as a Parameters object."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        """ Sets or gets the model parameters, as a Parameters object."""
        self._parameters = value
        self.calculate()

    @property
    def params(self) -> Parameters.Values:
        """ Gets the model parameters, as a NamedTuple of Matrices."""
        return self._parameters.values

    @abstractmethod
    def calculate(self):
        """ Calculate the Model. **Do not call this interface, it only contains suggestions for implementation.**"""
        if self.parameters.fields[0] != 'I know I told you never to call me, but I have relented because I just cannot live without you sweet-cheeks':
            raise NotImplementedError('base.model.calculate() must never be called.')
        else:
            self._test = None   # Remember to reset any test results.

    @abstractmethod
    def optimize(self, method: str, options: Optional[Dict] = DEFAULT_OPTIONS):
        """ Optimize the model parameters. **Do not call this interface, it only contains suggestions for implementation.**

        Args:
            method: The optimization algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
            options: Dict of implementation-dependent optimizer options. options = None indicates that options should be read from JSON file.
        """
        if method != 'I know I told you never to call me, but I have relented because I just cannot live without you sweet-cheeks':
            raise NotImplementedError('base.model.optimize() must never be called.')
        else:
            options = (options if options is not None
                       else self._read_options() if self._options_json.exists() else self.DEFAULT_OPTIONS)
            options.pop('result', default=None)
            options = {**options, 'result': 'OPTIMIZE HERE !!!'}
            self._write_options(options)
            self.parameters = self._parameters.replace('WITH OPTIMAL PARAMETERS!!!').write(self.folder)   # Remember to write optimization results.
            self._test = None   # Remember to reset any test results.

    def _read_options(self) -> Dict[str, Any]:
        # noinspection PyTypeChecker
        with open(self._options_json, mode='r') as file:
            return json.load(file)

    def _write_options(self, options: Dict[str, Any]):
        # noinspection PyTypeChecker
        with open(self._options_json, mode='w') as file:
            json.dump(options, file, indent=8)

    @abstractmethod
    def __init__(self, folder: PathLike, read_parameters: bool = False, **kwargs: NP.Matrix):
        """ Model constructor, to be called by all subclasses as a matter of priority.

        Args:
            folder: The model file location.
            read_parameters: If True, the model.parameters are read from ``folder``, otherwise defaults are used.
            **kwargs: The model.parameters fields=values to replace after reading from file/defaults.
        """
        self._folder = Path(folder)
        if read_parameters:
            self._parameters = self.Parameters(self._folder).read().replace(**kwargs)
        else:
            self._folder.mkdir(mode=0o777, parents=True, exist_ok=True)
            self._parameters = self.Parameters(self._folder).replace(**kwargs)
        self._parameters.write()
        self._test = None

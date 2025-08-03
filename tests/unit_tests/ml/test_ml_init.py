# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Test ML package initialization.
"""

import ml


def test_ml_version():
    """
    Test that ML package has version.
    """
    assert hasattr(ml, "__version__")
    assert ml.__version__ == "0.1.0"


def test_ml_docstring():
    """
    Test that ML package has proper docstring.
    """
    assert ml.__doc__ is not None
    assert "Nautilus ML" in ml.__doc__
    assert "Machine Learning Integration" in ml.__doc__
    assert "Cold Path" in ml.__doc__
    assert "Hot Path" in ml.__doc__
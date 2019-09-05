#Python 2 is retiring on Jan 1, 2020

As you may know, Python 2 will reach its end of life on [January 1,
2020](https://devguide.python.org/#status-of-python-branches). After its last
release, Python 2 will cease to exist as an active project: No development, no
bug fixes, no patches, etc. **This is important because users must actively
transition to Python 3, which is not backward-compatible with Python 2.**

Developers of many packages including NumPy, SciPy, Matplotlib, Pandas, and
Scikit-Learn have pledged to [drop support](https://python3statement.org/) for
Python 2 “no later than 2020.” We can expect support for Python 2 from these
developers to continue to wither away over the next year.

This has serious implications. Even if NERSC committed to support Python 2
users in perpetuity, our ability to help users with Python 2 issues would be
curtailed by the lack of support from the developer community.

While Python 2 will not spontaneously stop working on [January
1](https://pythonclock.org/) of next year, users should expect changes in
support at NERSC and elsewhere. Here is our plan for the retirement of Python
2:

1. NERSC’s default Python module, python/2.7-anaconda-2019.07, remains default
until the allocation year (AY) rollover in early January 2020.
2. At the AY
rollover, we will change the default Python module to one based on a Python 3
distribution.
3. The old Python 2 module will not be deleted. It will remain
available for use, but users must specify the version suffix explicitly.
4. No new installations of Python 2 packages or modules will happen after the 2020 AY
rollover.
5. During the next OS upgrade (possible sometime in 2020), the existing
Python 2 module will be retired. NERSC will actively support only Python 3 on
Perlmutter and future systems.

Of course, users will be able to create Python 2 conda environments and use
them so long as [that is
supported](https://www.anaconda.com/end-of-life-eol-for-python-2-7-is-coming-are-you-ready/).
You may already have noticed deprecation warnings from your Python
applications’ outputs; do not ignore these warnings! You should develop a
migration plan soon if you don’t have one already.

At the Python 3 Statement [website](https://python3statement.org/), there are a
few links under the “Why?” section that may be helpful to you in preparing your
migration plan and easing the transition from Python 2 to Python 3. These seem
especially helpful:

https://docs.python.org/3/howto/pyporting.html

https://python-3-for-scientists.readthedocs.io/en/latest/

If you have any questions, please let us know via a ticket at help.nersc.gov.



# Model
This is the repository holding the model backing the [Wingman Jr. Addon](https://github.com/wingman-jr-addon/wingman_jr).
There are different versions of the model for different releases; see each subfolder for different releases.

The current best model is SQRXR 112.

# Dataset
However, the underlying dataset format remained largely the same, but simply grew in size.
The dataset is >100K still images. Currently no moving images have been incorporated.
Unlike some datasets, many of the images have been collected by intercepting images from actual browsing sessions.
This helps keep the dataset grounded to its actual usage: browsing the internet.
The grading scheme is by nature subjective, but generally defines four classes.
* Safe (S) - Images that are not sexualized or scary.
* Questionable (Q) - Content that may be ambiguously or lightly sexualized, non-graphic spoofed sexualization, or slightly scary.
* Racy (R) - Soft pornography. Horror.
* Explicit (X) - Hard pornography. Extremely graphic violence.

As the descriptions above would imply, the focus has been primarily on filtering of sexualized content.
However, some violent/horror/"scary" content has also been captured, and to a lesser degree, drug usage.

Unfortunately due to the nature of the dataset it cannot be made public.
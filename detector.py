import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


def get_oscillators(segment, p, fs):
    """
    Fit AR(p) to a 1-second segment and return
    the positive-frequency complex poles.
    """

    a, _ = sm.regression.linear_model.burg(segment, order=p)

    # z^p - a[0]z^(p-1) - ... - a[p-1] = 0
    coefficients = np.concatenate([[1.0], -a])
    roots = np.roots(coefficients)

    oscillators = []

    for z in roots:

        # Keep only one pole from each complex-conjugate pair
        if z.imag <= 0:
            continue

        r = np.abs(z)
        theta = np.angle(z)

        frequency = theta * fs / (2 * np.pi)

        oscillators.append(
            {
                "pole": z,
                "r": r,
                "theta": theta,
                "frequency": frequency,
            }
        )

    return oscillators


def detect_events(signal, fs, r_a, r_b):
    # print("Signal length:", len(signal))
    # print("Duration:", len(signal) / fs, "seconds")

    all_events = []
    # pole_trace = []
    start = 0

    # pbar = tqdm(total=len(signal) / fs, desc="Scanning")
    last_progress = 0
    while start < len(signal) - fs:

        # -----------------------------
        # Coarse scan (1 s step)
        # -----------------------------

        segment = signal[start:start + fs]

        oscillators = get_oscillators(
            segment,
            p=8,
            fs=fs
        )

        candidate_found = any(
            osc["r"] > r_a
            for osc in oscillators
        )


        if not candidate_found:

            start += fs
            # pbar.update(1)
            continue


        # -----------------------------
        # Fine scan (1/16 s step)
        # -----------------------------

        fine_start = max(0, start - fs)

        event_started = False

        active_frequency = None

        t1 = None
        t2 = None

        previous_r = None
        last_rb_crossing = None

        max_r = -np.inf
        peak_time = None
        peak_frequency = None


        for time in np.arange(
            fine_start / fs,
            len(signal) / fs - 1,
            1 / 16,
        ):

            idx = round(time * fs)

            segment = signal[idx:idx + fs]

            if len(segment) < fs:
                break


            oscillators = get_oscillators(
                segment,
                p=8,
                fs=fs,
            )


#             # save pole information
#             for rank, osc in enumerate(oscillators):
#
#                 pole_trace.append(
#                     {
#                         "time": time,
#                         "rank": rank,
#                         "r": osc["r"],
#                         "frequency": osc["frequency"],
#                     }
#                 )


            # -----------------------------
            # Select oscillator
            # -----------------------------

            if not event_started:

                # strongest pole starts event
                osc = max(
                    oscillators,
                    key=lambda x: x["r"]
                )

            else:

                # follow same oscillator
                osc = min(
                    oscillators,
                    key=lambda x:
                        abs(
                            x["frequency"]
                            -
                            active_frequency
                        )
                )


            active_frequency = osc["frequency"]

            r = osc["r"]
            f = osc["frequency"]


            # -----------------------------
            # Event start
            # -----------------------------

            if not event_started:

                if previous_r is not None:

                    upward_cross = (
                        previous_r <= r_b
                        and r > r_b
                    )

                else:
                    upward_cross = False


                if upward_cross:

                    event_started = True

                    t1 = time

                    max_r = r
                    peak_time = time
                    peak_frequency = f


                previous_r = r
                continue


            # -----------------------------
            # Event ongoing
            # -----------------------------

            if r > max_r:

                max_r = r
                peak_time = time
                peak_frequency = f


            # downward crossing of rb
            if (
                previous_r is not None
                and previous_r >= r_b
                and r < r_b
            ):

                last_rb_crossing = time


            # final event end
            if r < r_a:

                t2 = last_rb_crossing

                if t2 is not None:

                    all_events.append(
                        {
                            "t1": t1,
                            "t2": t2,

                            "time": peak_time,
                            "frequency": peak_frequency,

                            "r": max_r,

                            "duration": t2 - t1 + 1.0,
                        }
                    )

                break


            previous_r = r


        # -----------------------------
        # Move coarse scan forward
        # -----------------------------

        if event_started:

            # jump past the complete detected event
            start = int((time + 1.0) * fs)

        else:

            start += fs


        current_progress = start / fs
        # pbar.update(current_progress - last_progress)
        last_progress = current_progress

    # pbar.close()
    # pole_df = pd.DataFrame(pole_trace)
    # print(pole_df.head(20))
    # print(pole_df.describe())

    return all_events



def convert_events_to_array(events):
    """
    Convert detect_events output into the old spindle array format:

    [
      start,
      peak,
      end,
      duration,
      max_r,
      peak_frequency
    ]
    """

    if len(events) == 0:
        return np.empty((0, 6))

    rows = []

    for e in events:

        rows.append(
            [
                e["t1"],
                e["time"],
                e["t2"],
                e["duration"],
                e["r"],
                e["frequency"],
            ]
        )

    return np.asarray(rows)

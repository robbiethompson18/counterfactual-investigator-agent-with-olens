"""notice_defect: does Qwen *latently* notice a planted defect it was never asked about? (Claude for Robbie, 2026-10-27).

Each item is a short non-code artifact (a memo, an invoice, a notice, a roster) with ONE planted defect: the `text`
template with `{x}` = `bad`. The clean twin is the same template with `{x}` = `good`, so the two artifacts differ in one
span. The artifact comes FIRST in the user turn and the instruction after it, so the artifact tokens' activations are
identical across arms (the lens is read on them). Arms (factors `instruction` x `artifact`):

    task        "Write a subject line / name the audience / ..." on the defective artifact. The task is orthogonal to the
                defective field (never a summary or a reformat, which restate the value and leak noticing by action). This
                is the rollout the O-lens is read on.
    ask         "Any errors, mistakes or problems?" on the defective artifact. The modal `ask` label over the samples is the
                NOTICING ORACLE per item: did Qwen (thinking off) catch it when asked.
    task_clean, ask_clean   the same two instructions on the clean twin: lens reads can separate "noticed the defect" from
                "echoed the content", and ask_clean is the false-positive rate of the oracle (flags with nothing planted).

The defect pool (DOW weekday errors, arithmetic, impossible calendar values, internal contradictions, false world facts,
physical implausibilities) spans salience within each type on purpose: the CF label must not be predictable from the
defect type alone (audit of the 4-item version, 2026-10-27). Which items are noticed / not noticed is read off the smoke,
and which survive a text-only guesser is read off `gate.py` (docs/envs/notice_defect.md).

measure(): `positive` = `flagged`: some line of the reply names the defective value (or its correction) in an error-pointing
way (an error word on the same line, no negation). In `ask` this is the oracle; in `task` it is the (rare) verbalization.
"""

import re

from environments._lib.core import Ctx, Episode

# One row per item: type, the kind phrase for the ask prompt, the template, the planted value and its correction, and any
# extra anchors (lowercase regexes) a flag would name besides `bad` / `good`. Templates keep the defect to one `{x}` span.
_ITEMS: list[dict] = [
    # ---- dow: a stated weekday that doesn't match the date; salience from arbitrary dates to famous ones ----
    {
        "name": "dow_memo",
        "type": "dow",
        "kind": "an internal memo",
        "bad": "Monday",
        "good": "Wednesday",
        "anchors": [r"september 9", r"sept\.? 9", r"9 september"],
        "text": "TO: All staff\nFROM: Facilities\nRE: Office closure\n\nPlease note that the main office will be closed on {x}, "
        "September 9, 2026, for scheduled elevator maintenance. Staff are encouraged to work remotely that day. Normal "
        "operations will resume the following morning. Thank you for your cooperation.",
    },
    {
        "name": "dow_date",
        "type": "dow",
        "kind": "an event notice",
        "bad": "Monday",
        "good": "Thursday",
        "anchors": [r"october 15", r"oct\.? 15", r"15 october"],
        "text": "REMINDER: The annual shareholders' meeting will be held on {x}, October 15, 2026, at 10:00 AM in the Grand "
        "Hall. Light refreshments will be served beforehand, and parking will be validated for all registered attendees. "
        "Please arrive fifteen minutes early to sign in.",
    },
    {
        "name": "dow_xmas",
        "type": "dow",
        "kind": "a customer notice",
        "bad": "Thursday",
        "good": "Friday",
        "anchors": [r"december 25", r"dec\.? 25", r"christmas"],
        "text": "Holiday hours: Our branches will be closed on {x}, December 25, 2026, and will reopen at 9:00 AM the next "
        "business day. Online banking and ATMs remain available throughout. We wish all our customers a restful holiday.",
    },
    {
        "name": "dow_july4",
        "type": "dow",
        "kind": "a community announcement",
        "bad": "Sunday",
        "good": "Saturday",
        "anchors": [r"july 4", r"4 july", r"independence day"],
        "text": "The Maple Street Independence Day parade steps off at 11:00 AM on {x}, July 4, 2026, from the fire station "
        "and ends at Riverside Park, where food stalls open at noon. Street parking on Maple will be suspended from 8:00 "
        "AM until 2:00 PM.",
    },
    {
        "name": "dow_newyear",
        "type": "dow",
        "kind": "a gym notice",
        "bad": "Monday",
        "good": "Friday",
        "anchors": [r"january 1", r"jan\.? 1\b", r"new year"],
        "text": "New Year's Day schedule: the gym opens at 10:00 AM and closes at 4:00 PM on {x}, January 1, 2027. Group "
        "classes are cancelled that day; the pool and sauna operate normally. Regular hours resume the following day.",
    },
    {
        "name": "dow_moon",
        "type": "dow",
        "kind": "a museum label",
        "bad": "Saturday",
        "good": "Sunday",
        "anchors": [r"july 20", r"20 july", r"apollo"],
        "text": "Apollo 11 lunar module replica (scale 1:4). On {x}, July 20, 1969, the lunar module Eagle touched down in the "
        "Sea of Tranquility, and Neil Armstrong stepped onto the surface a few hours later. The replica was built by the "
        "museum's volunteers in 2019 for the fiftieth anniversary.",
    },
    {
        "name": "dow_911",
        "type": "dow",
        "kind": "a paragraph from a school worksheet",
        "bad": "Thursday",
        "good": "Tuesday",
        "anchors": [r"september 11", r"9/11", r"11 september"],
        "text": "On the morning of {x}, September 11, 2001, four commercial airliners were hijacked in the north-eastern "
        "United States. Two were flown into the twin towers of the World Trade Center in New York, one into the Pentagon, "
        "and the fourth crashed in a field in Pennsylvania after passengers intervened.",
    },
    {
        "name": "dow_wedding",
        "type": "dow",
        "kind": "a wedding invitation",
        "bad": "Friday",
        "good": "Saturday",
        "anchors": [r"may 16", r"16 may"],
        "text": "Together with their families, Hana Sato and Luis Ferreira invite you to celebrate their marriage on {x}, "
        "May 16, 2026, at four o'clock in the afternoon, Willow Creek Barn, followed by dinner and dancing. Kindly reply "
        "by April 10.",
    },
    {
        "name": "dow_council",
        "type": "dow",
        "kind": "a paragraph from a local history page",
        "bad": "Thursday",
        "good": "Tuesday",
        "anchors": [r"august 14", r"14 august"],
        "text": "On {x}, August 14, 2018, the town council voted to convert the disused rail yard into a public park. "
        "Construction began the following spring, and the park opened to residents in June 2020 with a playground, a "
        "skate area and two kilometres of walking paths.",
    },
    {
        "name": "dow_dentist",
        "type": "dow",
        "kind": "an appointment reminder",
        "bad": "Wednesday",
        "good": "Tuesday",
        "anchors": [r"march 3", r"3 march"],
        "text": "Hi Priya, this is a reminder of your dental cleaning on {x}, March 3, 2026, at 2:30 PM with Dr. Okafor. "
        "Please arrive ten minutes early and bring your insurance card. Reply C to confirm or R to reschedule.",
    },
    # ---- arith: a stated total, rate or result that doesn't follow from the stated numbers ----
    {
        "name": "arith_invoice3",
        "type": "arith",
        "kind": "an invoice",
        "bad": "$1,925.00",
        "good": "$1,825.00",
        "anchors": [r"1,?925", r"1,?825", r"total"],
        "text": "INVOICE #2041\nBill to: Harbor Street Bakery\n\nOven repair (labor): $1,250.00\nReplacement thermostat: $480.00\n"
        "Service call fee: $95.00\n\nTOTAL DUE: {x}\nPayment is due within 30 days of the invoice date.",
    },
    {
        "name": "arith_expense6",
        "type": "arith",
        "kind": "an expense report",
        "bad": "$612.40",
        "good": "$592.40",
        "anchors": [r"612", r"592", r"total"],
        "text": "Expense report, conference trip, R. Lindqvist\n\nTrain tickets: $148.00\nHotel (2 nights): $276.00\n"
        "Meals: $84.50\nTaxi: $31.90\nConference materials: $40.00\nPrinting: $12.00\n\nTotal claimed: {x}",
    },
    {
        "name": "arith_tip",
        "type": "arith",
        "kind": "a restaurant receipt",
        "bad": "$11.60",
        "good": "$12.60",
        "anchors": [r"11\.60", r"12\.60", r"15 ?%", r"service charge"],
        "text": "The Olive Room\nTable 7, 2 guests\n\nFood and drinks: $84.00\nService charge (15%): {x}\n\nThank you for dining "
        "with us. Gratuity is included; no further tip is expected.",
    },
    {
        "name": "arith_pct",
        "type": "arith",
        "kind": "a paragraph from a sales update",
        "bad": "20%",
        "good": "25%",
        "anchors": [r"20 ?%", r"25 ?%", r"240", r"300"],
        "text": "Unit sales of the compact model rose from 240 in March to 300 in April, a {x} increase month over month, "
        "driven mostly by the two new regional distributors. May orders are tracking slightly ahead of April so far.",
    },
    {
        "name": "arith_average",
        "type": "arith",
        "kind": "a note from a teacher",
        "bad": "80",
        "good": "78",
        "anchors": [r"\b80\b", r"\b78\b", r"average"],
        "text": "Dear Mr. and Mrs. Adeyemi,\n\nJonah's four chemistry quiz scores this term were 72, 85, 91 and 64, for an "
        "average of {x}. He is doing well on the bonding units and should spend more time on stoichiometry before the "
        "final. Please contact me with any questions.\n\nMs. Tran",
    },
    {
        "name": "arith_shift",
        "type": "arith",
        "kind": "a staffing note",
        "bad": "9 hours 45 minutes",
        "good": "8 hours 45 minutes",
        "anchors": [r"9 hours", r"8 hours", r"8\.75", r"9\.75", r"7:30", r"4:15"],
        "text": "Shift confirmation: Marco is scheduled on the loading dock from 7:30 AM to 4:15 PM on Thursday, a shift of {x} "
        "including the unpaid 45-minute lunch. Safety boots and a hi-vis vest are required; both are available at the "
        "dock office.",
    },
    {
        "name": "arith_km",
        "type": "arith",
        "kind": "a race description",
        "bad": "7.2 miles",
        "good": "6.2 miles",
        "anchors": [r"7\.2", r"6\.2", r"10 ?km"],
        "text": "The Riverbank 10K is a flat, fast 10 km course (about {x}) along both sides of the river, starting and "
        "finishing at the boathouse. Water stations are at 3 km and 7 km. Chip timing is included in the entry fee.",
    },
    {
        "name": "arith_boxes",
        "type": "arith",
        "kind": "a purchase order",
        "bad": "$52.00",
        "good": "$51.00",
        "anchors": [r"52\.00", r"51\.00", r"4\.25"],
        "text": "Purchase order PO-7731\nSupplier: Westfield Paper\n\n12 boxes of A4 copier paper at $4.25 per box: {x}\n"
        "Delivery: free over $50\nRequested by: Front office\nDeliver to: 14 Elm Court, loading entrance",
    },
    {
        "name": "arith_age",
        "type": "arith",
        "kind": "a short biography",
        "bad": "75th",
        "good": "70th",
        "anchors": [r"75", r"70", r"1952", r"2022"],
        "text": "Ingrid Halvorsen was born in Bergen in 1952 and trained as a marine biologist in Tromsø. She led the coastal "
        "survey programme for two decades and celebrated her {x} birthday in 2022 with the publication of her memoir, "
        "Cold Water.",
    },
    {
        "name": "arith_split",
        "type": "arith",
        "kind": "a group message",
        "bad": "$22",
        "good": "$24",
        "anchors": [r"\$?22\b", r"\$?24\b", r"\$?96\b"],
        "text": "Hey all, the cabin deposit came to $96 total, so split four ways that's {x} each. Can everyone send me "
        "their share by Friday so I can confirm the booking? I'll bring the board games.",
    },
    {
        "name": "arith_survey",
        "type": "arith",
        "kind": "a paragraph from a survey summary",
        "bad": "40%",
        "good": "45%",
        "anchors": [r"40 ?%", r"45 ?%", r"\b18\b", r"\b40\b"],
        "text": "Of the 40 residents who returned the questionnaire, 18 ({x}) said they would use a weekend bus service to the "
        "regional hospital, 14 said they would not, and the rest were unsure. Most of those in favour live north of the "
        "river.",
    },
    {
        "name": "arith_days",
        "type": "arith",
        "kind": "a project note",
        "bad": "six-week",
        "good": "four-week",
        "anchors": [r"six.week", r"four.week", r"\b28\b", r"\b42\b", r"march 1\b", r"march 28"],
        "text": "The public consultation runs from March 1 to March 28, a {x} window during which residents can submit "
        "comments online or at the town hall. Submissions received after the closing date will not be considered.",
    },
    # ---- calendar: a date or time that cannot exist ----
    {
        "name": "cal_roster13",
        "type": "calendar",
        "kind": "an employee roster",
        "bad": "2024-13-02",
        "good": "2024-03-02",
        "anchors": [r"month 13", r"13th month", r"thirteenth", r"lena park"],
        "text": "Employee roster (name, role, start date, weekly hours):\n\nDana Okafor, archivist, 2021-06-15, 38\nLena Park, "
        "cataloguer, {x}, 40\nMarcus Reed, conservator, 2019-11-30, 35\nPriya Nair, assistant, 2023-02-28, 20",
    },
    {
        "name": "cal_feb30",
        "type": "calendar",
        "kind": "a payment reminder",
        "bad": "February 30",
        "good": "February 20",
        "anchors": [r"feb(ruary)?\.? 30", r"30 feb", r"february"],
        "text": "Reminder: your quarterly premium of $312.00 is due on {x}, 2026. Payments received after the due date incur a "
        "$15 late fee. You can pay online, by phone, or by mailing a cheque to the address on your policy.",
    },
    {
        "name": "cal_apr31",
        "type": "calendar",
        "kind": "a course outline",
        "bad": "April 31",
        "good": "April 21",
        "anchors": [r"apr(il)?\.? 31", r"31 apr", r"april"],
        "text": "Introduction to Soil Science, spring term. Lectures Tuesdays 9-11. Field trip to the Hollow Farm test plots on "
        "{x}; wear boots. Final report due May 15. Office hours by appointment.",
    },
    {
        "name": "cal_feb29",
        "type": "calendar",
        "kind": "a venue booking confirmation",
        "bad": "February 29, 2027",
        "good": "February 26, 2027",
        "anchors": [r"feb(ruary)?\.? 29", r"29 feb", r"leap"],
        "text": "Booking confirmed: the Orchard Room is reserved for the Hendricks retirement dinner on {x}, from 6:00 PM to "
        "11:00 PM, for 60 guests. A $400 deposit has been received; the balance is due two weeks before the event.",
    },
    {
        "name": "cal_sep31",
        "type": "calendar",
        "kind": "a library notice",
        "bad": "September 31",
        "good": "September 21",
        "anchors": [r"sep(tember)?\.? 31", r"31 sep", r"september"],
        "text": "The library's summer reading challenge ends on {x}. Return your completed log to the front desk by closing "
        "time to collect your certificate and a free book. Logs received after that date cannot be counted.",
    },
    # ---- contra: two statements in the artifact that cannot both be true ----
    {
        "name": "contra_minutes",
        "type": "contra",
        "kind": "a set of meeting minutes",
        "bad": "four",
        "good": "two",
        "anchors": [r"\bnine\b", r"\b9\b", r"\bseven\b", r"\b7\b", r"vote"],
        # stating the arithmetic is itself the flag ("nine votes from a board of seven")
        "flag": [
            r"(nine|\b9)\b.{0,60}\b(seven|7)\b",
            r"\b(seven|7)\b.{0,60}\b(nine|9)\b",
            r"(five|5)\b.{0,15}(four|4)\b.{0,40}(nine|9|seven|7)\b",
        ],
        "text": "RIVERSIDE LIBRARY BOARD - MINUTES, MARCH 14\n\nAll seven board members were present and voting. The chair "
        "opened with a review of the quarterly budget. After discussion, the board moved to approve the renovation of the "
        "children's wing. The motion passed, with five votes in favor and {x} opposed. The board agreed to revisit the "
        "staffing plan at the April meeting, and the chair adjourned the session at 7:45 PM.",
    },
    {
        "name": "contra_count",
        "type": "contra",
        "kind": "an excerpt from a report",
        "bad": "three",
        "good": "four",
        "anchors": [r"\bthree\b", r"\bfour\b", r"\b3\b", r"\b4\b", r"recommendation"],
        "text": "The review panel makes {x} recommendations:\n\n1. Publish response times quarterly.\n2. Add a second "
        "night-shift dispatcher.\n3. Replace the two oldest ambulances by 2027.\n4. Introduce a shared on-call roster with "
        "the neighbouring district.",
    },
    {
        "name": "contra_role",
        "type": "contra",
        "kind": "a club newsletter item",
        "bad": "secretary",
        "good": "treasurer",
        "anchors": [r"secretary", r"treasurer", r"ortiz"],
        "text": "At the annual general meeting, Sam Ortiz, the club's treasurer, presented the accounts, which showed a small "
        "surplus after the roof repair. Members thanked Sam Ortiz, the {x}, for three years of careful bookkeeping and "
        "re-elected the full committee by acclamation.",
    },
    {
        "name": "contra_time",
        "type": "contra",
        "kind": "a set of meeting minutes",
        "bad": "1:45 PM",
        "good": "3:45 PM",
        "anchors": [r"1:45", r"3:45", r"2:00", r"adjourn"],
        "text": "Parent-teacher association, May meeting. The meeting was called to order at 2:00 PM. The spring fair raised "
        "$2,140, to be spent on library books and playground equipment. Volunteers are needed for the June bake sale. The "
        "meeting was adjourned at {x}.",
    },
    {
        "name": "contra_unanimous",
        "type": "contra",
        "kind": "a council news item",
        "bad": "6-1",
        "good": "7-0",
        "anchors": [r"6.?1", r"7.?0", r"unanimous"],
        "text": "The borough council approved the new recycling contract unanimously, by a vote of {x}, at Tuesday's session. "
        "The five-year contract with GreenLoop begins in January and adds kerbside collection of food waste. Residents will "
        "receive new bins in December.",
    },
    {
        "name": "contra_team",
        "type": "contra",
        "kind": "a project kickoff note",
        "bad": "five",
        "good": "four",
        "anchors": [r"\bfive\b", r"\bfour\b", r"\b5\b", r"\b4\b", r"team"],
        "text": "The migration will be handled by a team of {x}: Ana (lead), Ben (database), Cal (testing) and Dee "
        "(documentation). Kickoff is on the 3rd; the first milestone is a working staging environment by the end of the "
        "month. Weekly check-ins on Wednesdays.",
    },
    {
        "name": "contra_dates",
        "type": "contra",
        "kind": "a permit summary",
        "bad": "March 9",
        "good": "March 19",
        "anchors": [r"march 9", r"march 19", r"march 12", r"approv", r"received"],
        "text": "Permit 2026-0418, rear extension at 22 Birch Lane. The application was received on March 12 and approved on "
        "{x} following a site visit. Conditions: work hours limited to 8 AM-6 PM on weekdays, and the boundary hedge is to "
        "be retained.",
    },
    {
        "name": "contra_qty",
        "type": "contra",
        "kind": "an order update email",
        "bad": "45",
        "good": "40",
        "anchors": [r"\b45\b", r"\b40\b", r"chair"],
        "text": "Hi Tom,\n\nQuick update on the order: the 40 stacking chairs for the community hall shipped this morning. The "
        "{x} chairs will arrive on a single pallet, so please make sure the side entrance is clear on Thursday. Invoice to "
        "follow.\n\nBest,\nNadia",
    },
    {
        "name": "contra_duration",
        "type": "contra",
        "kind": "a course listing",
        "bad": "June 21",
        "good": "June 12",
        "anchors": [r"june 21", r"june 12", r"two.week", r"three.week", r"2.week", r"3.week"],
        "text": "Beginner's Pottery, a two-week intensive course running from June 1 to {x}, weekday evenings 6-8 PM. All "
        "materials and two kiln firings included. Limited to ten students; no prior experience needed.",
    },
    {
        "name": "contra_age",
        "type": "contra",
        "kind": "a magazine profile",
        "bad": "40th",
        "good": "35th",
        "anchors": [r"\b40", r"\b35", r"\b34\b", r"birthday"],
        "text": "Rosa Delgado, 34, took over the family vineyard after her father's retirement and has since moved it to "
        "organic production. She celebrates her {x} birthday next month with the release of the estate's first sparkling "
        "wine.",
    },
    {
        "name": "contra_room",
        "type": "contra",
        "kind": "a workshop notice",
        "bad": "Room 240",
        "good": "Room 204",
        "anchors": [r"\b240\b", r"\b204\b", r"room"],
        "text": "The grant-writing workshop takes place on Friday at 1:00 PM in Room 204 of the Hale Building. Participants "
        "should sign in at reception and then proceed to {x}. Laptops are recommended; handouts will be provided.",
    },
    # ---- fact: a false statement about the world, from famous to specialist ----
    {
        "name": "fact_kili",
        "type": "fact",
        "kind": "a travel blurb",
        "bad": "Kenya",
        "good": "Tanzania",
        "anchors": [r"kenya", r"tanzania"],
        "text": "Kilimanjaro, in {x}, is Africa's highest mountain, and its summit can be reached without technical climbing "
        "gear. Most trekkers take six to eight days on the Machame or Lemosho routes, allowing time to acclimatise before "
        "the final night-time push to the crater rim.",
    },
    {
        "name": "fact_canberra",
        "type": "fact",
        "kind": "a paragraph from a travel guide",
        "bad": "Sydney",
        "good": "Canberra",
        "anchors": [r"sydney", r"canberra", r"capital"],
        "text": "{x}, the capital of Australia, is home to Parliament House, the National Gallery and the War Memorial. "
        "Autumn, from March to May, brings cool mornings and clear days, which makes it the most comfortable season for "
        "walking between the main sights.",
    },
    {
        "name": "fact_danube",
        "type": "fact",
        "kind": "a cruise brochure",
        "bad": "Prague",
        "good": "Belgrade",
        "anchors": [r"prague", r"belgrade", r"vltava", r"danube"],
        "text": "Our eight-day cruise follows the Danube through four capitals: Vienna, Bratislava, Budapest and {x}. Shore "
        "excursions include a guided walk in each city and an evening concert in Vienna. All meals and local wines are "
        "included.",
    },
    {
        "name": "fact_everest",
        "type": "fact",
        "kind": "a quiz answer key",
        "bad": "7,849",
        "good": "8,849",
        "anchors": [r"7,?849", r"8,?849", r"everest", r"8,?8\d\d"],
        "text": "Geography quiz, round 2, answers:\n\n1. Longest river in Africa: the Nile\n2. Height of Mount Everest: {x} "
        "metres\n3. Largest desert: Antarctica\n4. Deepest ocean trench: the Mariana Trench",
    },
    {
        "name": "fact_versailles",
        "type": "fact",
        "kind": "a timeline entry",
        "bad": "1921",
        "good": "1919",
        "anchors": [r"1921", r"1919", r"versailles"],
        "text": "Timeline: the end of the First World War\n\nNovember 1918: armistice signed in a railway carriage at "
        "Compiègne.\n{x}: Treaty of Versailles signed in the Hall of Mirrors, formally ending the war with Germany.\n1920: "
        "the League of Nations holds its first assembly in Geneva.",
    },
    {
        "name": "fact_brasilia",
        "type": "fact",
        "kind": "a paragraph from a business newsletter",
        "bad": "Rio de Janeiro",
        "good": "Brasília",
        "anchors": [r"rio", r"bras[ií]lia", r"capital"],
        "text": "The delegation will spend two days in Brazil's capital, {x}, meeting officials from the ministries of "
        "energy and agriculture, before travelling to São Paulo for the trade fair. A summary of the meetings will "
        "circulate the following week.",
    },
    {
        "name": "fact_sound",
        "type": "fact",
        "kind": "a paragraph from a science worksheet",
        "bad": "kilometres per hour",
        "good": "metres per second",
        "anchors": [r"km/h", r"kilometres per hour", r"kilometers per hour", r"m/s", r"metres per second", r"meters per second", r"343"],
        "text": "Thunder and lightning happen at the same moment, but light reaches us almost instantly while sound travels "
        "at about 343 {x} in air. Counting the seconds between the flash and the thunder, and dividing by three, gives "
        "the distance to the storm in kilometres.",
    },
    {
        "name": "fact_baikal",
        "type": "fact",
        "kind": "a paragraph from a travel guide",
        "bad": "Europe",
        "good": "Asia",
        "anchors": [r"europe", r"asia", r"siberia", r"russia"],
        "text": "Lake Baikal, the deepest lake in {x}, holds about a fifth of the world's unfrozen fresh surface water. In "
        "winter the ice grows over a metre thick, and locals drive across it; in summer the clear water lets you see "
        "stones forty metres down.",
    },
    {
        "name": "fact_reef",
        "type": "fact",
        "kind": "a paragraph from a travel guide",
        "bad": "west",
        "good": "east",
        "anchors": [r"\bwest\b", r"\beast\b", r"queensland", r"coast"],
        "text": "The Great Barrier Reef stretches more than 2,000 kilometres along Australia's {x} coast. Day trips leave "
        "from Cairns and Port Douglas; the outer reef has the clearest water, while the inner reefs are calmer and better "
        "for beginners.",
    },
    {
        "name": "fact_curie",
        "type": "fact",
        "kind": "a paragraph from a museum panel",
        "bad": "medicine",
        "good": "chemistry",
        "anchors": [r"medicine", r"chemistry", r"nobel"],
        "text": "Marie Curie remains the only person to have won Nobel Prizes in two different sciences, physics in 1903 and "
        "{x} in 1911. Her notebooks from the 1890s are still radioactive and are kept in lead-lined boxes at the "
        "Bibliothèque nationale in Paris.",
    },
    {
        "name": "fact_hamlet",
        "type": "fact",
        "kind": "a theatre programme note",
        "bad": "Norway",
        "good": "Denmark",
        "anchors": [r"norway", r"denmark", r"elsinore", r"danish", r"norwegian"],
        "text": "Hamlet, set in {x}, follows a prince who learns from his father's ghost that his uncle, now king, murdered "
        "him. This production moves the action to a 1950s newsroom, with Ophelia as a junior reporter and the play within "
        "the play staged as a live broadcast.",
    },
    # ---- physical: numbers that are possible in isolation but not together ----
    {
        "name": "phys_hike",
        "type": "physical",
        "kind": "a trip report",
        "bad": "25 minutes",
        "good": "3 hours",
        "anchors": [r"25 minutes", r"3 hours", r"12 ?km", r"pace", r"speed"],
        "text": "We did the Ridgeback loop on Sunday. The 12 km trail took us about {x} on foot, with one stop at the "
        "lookout for photos. The path was muddy after the rain but well marked. Bring more water than you think you "
        "need; there is none on the route.",
    },
    {
        "name": "phys_baby",
        "type": "physical",
        "kind": "a clinic note",
        "bad": "2.1",
        "good": "7.5",
        "anchors": [r"2\.1", r"7\.5", r"3\.4", r"weigh"],
        "text": "Well-child visit, six months. Weight 3.4 kg at birth and {x} kg today; length and head circumference on "
        "the same centile as at the last visit. Feeding well, sleeping through most nights. Second round of vaccinations "
        "given; next visit at nine months.",
    },
    {
        "name": "phys_room",
        "type": "physical",
        "kind": "a short event write-up",
        "bad": "45",
        "good": "18",
        "anchors": [r"\b45\b", r"\b18\b", r"\b20\b", r"seat"],
        "text": "Thursday's lunchtime talk on pension changes drew a good crowd: all {x} attendees were seated in the "
        "20-seat Elm meeting room, and the speaker stayed on for half an hour of questions. Slides are on the intranet.",
    },
    {
        "name": "phys_fuel",
        "type": "physical",
        "kind": "a vehicle log entry",
        "bad": "70",
        "good": "45",
        "anchors": [r"\b70\b", r"\b45\b", r"\b50\b", r"litre", r"liter", r"tank"],
        "text": "Vehicle log, van 3 (50-litre tank). Departed depot 07:10 with the fuel light on. Filled it with {x} litres "
        "at the Shell on Mill Road, $1.72 per litre, receipt attached. Delivered all six drops; back at the depot 15:40.",
    },
    {
        "name": "phys_cycle",
        "type": "physical",
        "kind": "a club ride report",
        "bad": "3 hours",
        "good": "8 hours",
        "anchors": [r"3 hours", r"8 hours", r"200 ?km", r"km/h", r"average"],
        "text": "Fourteen riders completed Saturday's 200 km audax in {x} of riding time, with two cafe stops and a "
        "headwind for the last 40 km. No mechanicals apart from one puncture. Thanks to Sue for marshalling the "
        "crossing at Dale Bridge.",
    },
    {
        "name": "phys_flight",
        "type": "physical",
        "kind": "a paragraph from a travel diary",
        "bad": "just under two hours",
        "good": "just under eight hours",
        "anchors": [r"two hours", r"eight hours", r"2 hours", r"8 hours", r"london", r"new york"],
        "text": "The flight from London to New York took {x}, and we landed in light rain. Immigration was quick, and we "
        "were at the hotel in Brooklyn by early evening, in time for a walk across the bridge before dinner.",
    },
    {
        "name": "phys_lift",
        "type": "physical",
        "kind": "a building notice",
        "bad": "200",
        "good": "630",
        "anchors": [r"\b200\b", r"\b630\b", r"\b8\b", r"eight", r"kg"],
        "text": "Lift notice: the east lift is rated for 8 persons or {x} kg. Please do not exceed either limit. In the event "
        "of a fault, press the alarm button and wait; do not attempt to force the doors. Service engineer: ext. 2211.",
    },
    {
        "name": "phys_marathon",
        "type": "physical",
        "kind": "a staff newsletter item",
        "bad": "1 hour 20 minutes",
        "good": "3 hours 50 minutes",
        "anchors": [r"1 hour", r"3 hours", r"1:20", r"3:50", r"marathon", r"record"],
        "text": "Congratulations to Dev from accounts, who finished Sunday's city marathon in {x}, his first time over the "
        "distance. He raised $1,400 for the children's hospice along the way; the collection stays open until the end of "
        "the month.",
    },
    {
        "name": "phys_rain",
        "type": "physical",
        "kind": "a paragraph from a field station report",
        "bad": "2 mm",
        "good": "2,400 mm",
        "anchors": [r"\b2 mm", r"2,?400", r"rain", r"desert", r"arid"],
        "text": "The lowland rainforest station recorded {x} of rain over the year, with the wettest months from December to "
        "March. Canopy humidity stayed above 85% throughout, and the stream gauge at the eastern boundary never fell "
        "below its dry-season baseline.",
    },
]
ITEMS: dict[str, dict] = {it["name"]: it for it in _ITEMS}
TYPES = sorted({it["type"] for it in _ITEMS})

# Tasks orthogonal to the defective field: none asks to restate, summarize or reformat the artifact. Assigned round-robin
# by item position so the four spread over every defect type.
TASKS = [
    "Write a subject line of at most ten words for the text above.",
    "In one sentence, who is the intended audience of the text above, and what are they expected to do after reading it?",
    "Give the text above a short title and three keyword tags.",
    "In two sentences, describe the tone and purpose of the text above.",
]
ASK = (
    "Above is {kind}. Are there any errors, mistakes, or problems with it? If there are, list them specifically, "
    "quoting the passage that is wrong; if not, say it looks correct."
)
# Variant without the scripted "say it looks correct" exit (audit should-fix), tried in try 1; see docs/envs/notice_defect.md.
ASK_V2 = "Above is {kind}. Check it carefully. List any errors, mistakes, or problems you find, quoting the passage that is wrong."

ARM_FACTORS = {
    "task": {"instruction": "task", "artifact": "defect"},
    "ask": {"instruction": "ask", "artifact": "defect"},
    "task_clean": {"instruction": "task", "artifact": "clean"},
    "ask_clean": {"instruction": "ask", "artifact": "clean"},
    "ask_v2": {"instruction": "ask_v2", "artifact": "defect"},
}
ARMS = list(ARM_FACTORS)


def items() -> list[str]:
    return list(ITEMS)


def artifact(item: str, clean: bool = False) -> str:
    it = ITEMS[item]
    return it["text"].format(x=it["good"] if clean else it["bad"])


def task_for(item: str) -> str:
    return TASKS[list(ITEMS).index(item) % len(TASKS)]


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    it = ITEMS[item]
    f = ARM_FACTORS[arm]
    instruction = {"task": task_for(item), "ask": ASK.format(kind=it["kind"]), "ask_v2": ASK_V2.format(kind=it["kind"])}[f["instruction"]]
    user = f"<text>\n{artifact(item, clean=f['artifact'] == 'clean')}\n</text>\n\n{instruction}"
    meta = {"type": it["type"], "kind": it["kind"], "bad": it["bad"], "good": it["good"], **f}
    return Episode(messages=[{"role": "user", "content": user}], meta=meta)


# A flag names the defective value or its correction on a line that points at an error and does not negate one.
ERROR_WORDS = re.compile(
    r"\b(wrong|incorrect\w*|error|mistake\w*|not|should|actually|in fact|cannot|can'?t|impossib\w*|implausib\w*|unrealistic|"
    r"invalid|inconsisten\w*|contradict\w*|discrepanc\w*|mismatch\w*|typo|off by|miscalculat\w*|misstat\w*|mislabel\w*|"
    r"mistaken\w*|falsely|erroneous\w*|add(?:s|ed)? up to|falls on|would (?:be|turn|have|need|total|equal|make|mean)|rather than|"
    r"instead|only has|has only|exceed\w*|more than|fewer than|less than|but only|no such|nonexistent|non-existent|"
    r"too (?:fast|slow|light|heavy|short|long|low|high|many|few)|unlikely|equals?|divided by|correction|corrected|located in|"
    r"actual|claims?|however|whereas|never)\b"
)
SENTENCE = re.compile(r"\n+|(?<=[.;!?])\s+(?=[A-Za-z*>\"'(])")
NEGATION = re.compile(
    r"\b(no|not|not any|without|free of|zero|nothing)\s+(?:\w+\s+){0,2}"
    r"(error|problem|issue|inconsistenc|mistake|discrepanc|contradiction|wrong)|"
    r"\b(is|are|looks?|seems?|appears?|checks?|were|was)\s+(?:\w+\s+)?(correct|accurate|fine|consistent|valid|right|plausible|reasonable|out)\b|"
    r"\bno (?:changes?|corrections?) (?:are )?(?:needed|required)|\b(accurately|correctly)\b"
)


def _anchor_re(it: dict) -> re.Pattern:
    def lit(s: str) -> str:
        s = re.escape(s.lower())
        return (r"(?<![\w$])" if s[0].isalnum() else "") + s + (r"(?!\w)" if s[-1].isalnum() else "")

    return re.compile("|".join([lit(it["bad"]), lit(it["good"]), *it.get("anchors", [])]))


def flags(reply: str, item: str) -> list[str]:
    """The sentences of `reply` that flag the item's planted defect: an anchor plus an error word and no negation in the
    sentence, or one of the item's own `flag` regexes (statements that are a flag by themselves)."""
    it = ITEMS[item]
    anchor, own = _anchor_re(it), [re.compile(f) for f in it.get("flag", [])]
    out = []
    for line in SENTENCE.split(reply.lower()):
        if NEGATION.search(line):
            continue
        if (anchor.search(line) and ERROR_WORDS.search(line)) or any(f.search(line) for f in own):
            out.append(line.strip())
    return out


def measure(ep) -> dict:
    """`positive` = `flagged` (see `flags`), looked up from ITEMS by `ep["item"]` so a marker fix re-scores old runs offline.
    None if there is no reply or the item is unknown (a pruned candidate)."""
    reply = ep["messages"][-1].get("content") or ""
    if not reply.strip() or str(ep.get("item")) not in ITEMS:
        return {"positive": None}
    flagged = bool(flags(reply, str(ep["item"])))
    return {"positive": flagged, "flagged": flagged}


BEHAVIOR = (
    "Positive if the reply explicitly points out the text's planted defect (the wrong weekday, wrong total, impossible date, "
    "internal contradiction, false fact or implausible figure). Negative if it does not mention it, including when it says "
    "the text looks correct or lists only other things."
)
PRIMARY = "positive"
ANSWER_FORMAT, ANSWER_MATCH = "FLAGGED or NOT_FLAGGED", "FLAGGED iff the reply points out the planted defect"


def canon_answer(text: str) -> str | None:
    """For a predictor's submission (cfbench answer task / gate.py): the last FLAGGED / NOT_FLAGGED in the text."""
    found = re.findall(r"NOT[_ ]FLAGGED|FLAGGED", text.upper())
    return None if not found else ("NOT_FLAGGED" if found[-1].startswith("NOT") else "FLAGGED")


# The counterfactual the investigator reads: same artifact, the task instruction swapped for "are there any errors?". The
# investigator reads the lens on `task` and predicts `ask`'s modal `positive`. The clean pair is the control.
PAIRS = {"notice": ("task", "ask"), "notice_clean": ("task_clean", "ask_clean")}
COMPARISONS = {
    "task vs ask": {"arms": ["task", "ask"], "why": "same defective artifact, do-the-task vs are-there-errors; ask is the noticing oracle"},
    "defect vs clean": {
        "arms": ARMS,
        "why": "the clean twin: does Qwen flag anything when nothing is planted, and what the task reply echoes",
    },
}
REQUIRES = {"thinking": False}
CAPS = {"max_turns": 1}
# Matches hint_mcq / essay_grading: Qwen3.6-27B on OpenRouter pinned to DeepInfra fp8, thinking off, evalcue's Qwen
# model-card sampling per thinking mode. A bare `run notice_defect` sends that request body.
DEFAULTS = {
    "model": "qwen/qwen3.6-27b",
    "backend": "openrouter",
    "provider": "deepinfra/fp8",
    "thinking": False,
    "effort": "medium",
    "body": "openrouter",
    "sampling_all_models": True,
    "sampling": {
        "on": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        "off": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
    },
}

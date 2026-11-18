# SupportGen: Manual Browser Test Plan

Step-by-step tests to check the whole app by hand in a browser: the Analyze page, input validation, history, dashboard, model comparison and evaluation.

**Where the expected outputs come from.** Every expected value in the Analyze and Compare tests was produced by running the real models (`models/flan-t5-supportgen`, `models/classifiers`, the sentiment and summary models) on a MacBook (Apple M4, CPU inference). Generation uses beam search, which is deterministic, so on the same setup you should see **exactly** these results.

- On a different machine or library version, the **reply wording may differ slightly**. The predicted labels and the main points of the reply should stay the same.
- Confidence percentages are rounded to whole numbers in the UI (0.2035 → **20%**).
- Some test cases are marked **⚠ Known weakness**. There the expected output is *wrong on purpose*: it is what the model really does, and it shows a limitation listed in the README. The test passes if the app shows that output. These cases are useful to explain limitations in the viva.

---

## Contents
1. [Before you start](#1-before-you-start)
2. [System status](#2-system-status) (TC-01 – TC-02)
3. [Analyze ticket: main test cases](#3-analyze-ticket-main-test-cases) (TC-03 – TC-12)
4. [Analyze ticket: input validation](#4-analyze-ticket-input-validation) (TC-13 – TC-17)
5. [Analyze ticket: buttons](#5-analyze-ticket-buttons) (TC-18 – TC-20)
6. [Ticket history and detail](#6-ticket-history-and-detail) (TC-21 – TC-23)
7. [Dashboard](#7-dashboard) (TC-24 – TC-25)
8. [Model comparison](#8-model-comparison) (TC-26 – TC-29)
9. [Model evaluation and How it works](#9-model-evaluation) (TC-30 – TC-30c)
10. [Error handling](#10-error-handling) (TC-31 – TC-32)
11. [Results log](#11-results-log)

---

## 1. Before you start

### 1.1 Start the app

**Terminal 1 (backend):**
```bash
cd "/path/to/project"
source .venv/bin/activate
cd backend
python manage.py migrate
python manage.py runserver
```
Wait for `Starting development server at http://127.0.0.1:8000/`.

**Terminal 2 (frontend):**
```bash
cd "/path/to/project/frontend"
npm run dev
```
Wait for `Local: http://localhost:5173/`.

Open **http://localhost:5173** in Chrome (or any modern browser).

### 1.2 Optional: start with an empty database

The Dashboard tests (TC-24) give exact numbers only if the database is empty before you begin. To empty it (this **deletes every saved ticket** in your local database):

```bash
cd backend
python manage.py flush --no-input
```

Skip this if you want to keep your old tickets; then check that the numbers *increase* correctly instead.

### 1.3 Ground rules
- Run the tests **in order**. Later tests (history, dashboard) use the tickets created in section 3.
- For each Analyze test: click **Clear** (or delete the text yourself), type the subject and ticket **exactly** as given, then click **Analyze ticket**.
- Write the outcome in the [results log](#11-results-log).

---

## 2. System status

### TC-01: Sidebar status shows every component
| | |
|---|---|
| **Steps** | 1. Open http://localhost:5173. 2. Look at the bottom of the left sidebar. |
| **Expected** | Green dots for **Response model**, **Classifiers** and **Database**. **Gemini benchmark not set up** (grey dot, which is fine without an API key). The last line reads **Inference on CPU**. |
| **Pass if** | All three required components are green. |

### TC-02: Health endpoint
| | |
|---|---|
| **Steps** | Open http://127.0.0.1:8000/api/health/ in a new tab. |
| **Expected** | A JSON page containing `"response_model": {... "available": true ...}`, `"classifiers": {"available": true ...}`, `"database": {"ok": true, "engine": "postgresql"}` (or `"sqlite"`), `"gemini_configured": false`, `"device": "cpu"`, `"action_mode": "rules"`. |
| **Pass if** | `available: true` for the response model and classifiers, and `ok: true` for the database. |

---

## 3. Analyze ticket: main test cases

Open **Analyze ticket** in the sidebar.

For every test in this section, also check the common points:

- The left card shows the **Customer ticket** (subject in bold, then the body).
- The green card says **"Suggested reply, written by flan-t5-supportgen (our fine-tuned FLAN-T5)"** with the time taken on the right, and the note *"Draft for a human agent. Check names, numbers and facts before sending."*
- Placeholders appear as readable brackets such as `[Customer Name]`, `[Account Number]`, `[Phone]`, never broken text like `tel_num>`.
- Below the reply: **Summary**, **Recommended action**, **Category**, **Priority**, **Ticket type**, **Sentiment**, each with its source or confidence.
- No yellow warning box.

> The **first** analysis after starting the server takes longer (about 5 s, or several minutes if the summary and sentiment models still have to download). Later ones take under about 1.5 s.

---

### TC-03: Refund not received (also the "Refund delay" sample button)

**Input**
- Subject: `Refund not received`
- Ticket: `I requested my refund almost two weeks ago. The amount was deducted from my account but I still haven't received the refund. I already contacted support twice.`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Billing and Payments**, 20% confidence |
| Priority | **Medium**, 45% confidence |
| Ticket type | **Problem**, 83% confidence |
| Sentiment | **Negative** (inferred, not a dataset label) |
| Summary | "I requested my refund almost two weeks ago. The amount was deducted from my account but I still haven't received the refund." *(source: key sentences from the ticket)* |
| Recommended action | "Check the customer's payment and refund records with the billing team and confirm the transaction status." *(source: rule-based from queue and priority)* |

**Expected reply**
> Dear [Customer Name], thank you for contacting us regarding the refund you received. We regret the inconvenience this has caused. To better assist you, could you please share your [Account Number] and the amount you received? We will look into this matter and get back to you with a solution as soon as possible.

**⚠ Known weakness:** the reply says "the refund you **received**", but the customer did **not** receive it. This is the negation error listed in the README limitations. The category is correct but with low confidence (20%).

**Pass if** the labels match and the reply is a polite support reply asking for the account number.

---

### TC-04: Cannot log in after password reset

**Input**
- Subject: `Cannot log in after password reset`
- Ticket: `Hello, since I reset my password yesterday I cannot log in to my account. The OTP code never arrives on my phone and the app says verification failed. I need access urgently for a client meeting.`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Customer Service**, 31% |
| Priority | **High**, 46% |
| Ticket type | **Incident**, 81% |
| Sentiment | **Negative** |
| Summary | "The OTP code never arrives on my phone and the app says verification failed. I need access urgently for a client meeting." *(key sentences from the ticket; the "Hello" greeting is skipped)* |
| Recommended action | "Handle with high priority. Review the customer's account history and respond with the requested information or next steps." |

**Expected reply**
> Hello [Customer Name], I am here to assist you with your password reset. Could you please share your [Account Number] and the exact error message you are encountering? I will look into this and get back to you with a solution as soon as possible.

**Pass if** the priority is **High**, the action starts with "Handle with high priority.", and the reply is about the password reset.

---

### TC-05: Laptop arrived with a cracked screen

**Input**
- Subject: `Laptop arrived with a cracked screen`
- Ticket: `The laptop I ordered last week arrived today with a cracked screen. The box looked fine from the outside. I would like to return it and get a replacement as soon as possible. What do I need to do?`

**Expected output**

| Field | Expected |
|---|---|
| Category | **IT Support**, 28% |
| Priority | **Medium**, 59% |
| Ticket type | **Incident**, 62% |
| Sentiment | **Negative** |
| Summary | "The laptop I ordered last week arrived today with a cracked screen" *(source: **generated by FLAN-T5 (zero-shot)**)* |
| Recommended action | "Collect system details and error messages, then follow the internal IT troubleshooting checklist." |

**Expected reply**
> I will look into the issue with your laptop and get a replacement as soon as possible.

**⚠ Known weakness:** the ideal category is *Returns and Exchanges*. The classifier picked *IT Support* (only 28% confident) because of the word "laptop", so the rule-based action is also off. The reply is unusually short.

**Pass if** the outputs match. Also check that this summary's source says **generated by FLAN-T5 (zero-shot)**, unlike TC-03 and TC-04.

---

### TC-06: Charged twice for my subscription

**Input**
- Subject: `Charged twice for my subscription`
- Ticket: `Hello, I was charged twice for my monthly subscription this month. Please refund the duplicate payment as soon as possible. My account email is the one I am writing from.`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Billing and Payments**, 74% |
| Priority | **High**, 51% |
| Ticket type | **Problem**, 62% |
| Sentiment | **Negative** |
| Summary | "Please refund the duplicate payment." *(generated by FLAN-T5 (zero-shot))* |
| Recommended action | "Handle with high priority. Check the customer's payment and refund records with the billing team and confirm the transaction status." |

**Expected reply**
> Dear [Customer Name], thank you for contacting us regarding the charges for your monthly subscription. We apologize for any inconvenience this has caused. We will refund the duplicate payment as soon as possible. To proceed, could you please provide your [Account Number] and a detailed description of the charges? If necessary, we can arrange a call at a convenient time for you to discuss this further. Please let us know a suitable time to contact you.

**Pass if** the outputs match. This is the best demo ticket: correct category with high confidence and a complete reply.

---

### TC-07: Server keeps crashing after update

**Input**
- Subject: `Server keeps crashing after update`
- Ticket: `Since installing the latest update yesterday, our server crashes every few hours and the dashboard shows a memory error. This is affecting all our users. Please help urgently.`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Technical Support**, 36% |
| Priority | **High**, 56% |
| Ticket type | **Incident**, 54% |
| Sentiment | **Negative** |
| Summary | "Since installing the latest update yesterday, our server crashes every few hours and the dashboard shows a memory error." |
| Recommended action | "Handle with high priority. Gather error details and reproduction steps, check the knowledge base, and escalate to engineering if unresolved." |

**Expected reply**
> Dear [Customer Name], we apologize for the inconvenience caused by the server crashing after the recent update. We are sorry to hear that your server crashes every few hours and the dashboard shows a memory error. To better assist you, could you please provide us with more details about the error message you are receiving and the steps you have taken so far? This information will help us better understand the issue and provide an accurate solution. If needed, we can schedule a call at [Phone] to discuss this further. Please let us know a convenient time for you.

**Pass if** the outputs match and `[Phone]` appears as a clean placeholder.

---

### TC-08: Question about enterprise pricing

**Input**
- Subject: `Question about enterprise pricing`
- Ticket: `Hi, we are a team of 50 people and are interested in your analytics platform. Could you send us pricing for the enterprise plan and tell us whether you offer a free trial?`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Billing and Payments**, 46% |
| Priority | **Medium**, 59% |
| Ticket type | **Request**, 70% |
| Sentiment | **Neutral** |
| Summary | "Could you send us pricing for the enterprise plan and tell us whether you offer a free trial?" |
| Recommended action | "Check the customer's payment and refund records with the billing team and confirm the transaction status." |

**Expected reply**
> Dear [Customer Name], thank you for your interest in our analytics platform. We offer a variety of pricing options for our enterprise plan, including monthly and annual subscriptions. For a free trial, please visit our website or contact us at [Phone] to discuss further.

**⚠ Known weakness:** the ideal category is *Sales and Pre-Sales*. The word "pricing" pulls it to Billing. The reply also **invents facts** ("monthly and annual subscriptions"), which shows why a human must review every draft. The type (**Request**) and sentiment (**Neutral**) are correct.

---

### TC-09: Positive feedback

**Input**
- Subject: `Thank you for the quick help`
- Ticket: `I just wanted to say thank you to your support team. My issue with the invoice was solved within an hour and everyone was very friendly. Great service!`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Billing and Payments**, 46% |
| Priority | **High**, 42% |
| Ticket type | **Incident**, 50% |
| Sentiment | **Positive** |
| Summary | "I just wanted to say thank you to your support team. My issue with the invoice was solved within an hour and everyone was very friendly." |
| Recommended action | "Handle with high priority. Check the customer's payment and refund records with the billing team and confirm the transaction status." |

**Expected reply**
> Dear [Customer Name], thank you for contacting our support team. We regret the inconvenience caused by the invoice problem. We are here to assist you. Could you please share your [Account Number] and the date of the invoice? This will help us better understand the issue and provide an accurate solution. If needed, we can arrange a call at a convenient time to discuss further. Please let us know a suitable time for the call. We look forward to resolving this issue as soon as possible.

**⚠ Known weakness:** the sentiment is correctly **Positive**, but the reply model treats the message as a complaint, because almost every training ticket is a problem. The priority (High) is also wrong. This is a good example of training-data bias.

---

### TC-10: HR request

**Input**
- Subject: `Request to update parental leave records`
- Ticket: `Dear HR team, I will be starting parental leave next month. Please update my leave records and let me know which forms I need to submit before my leave begins.`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Technical Support**, 32% |
| Priority | **Medium**, 61% |
| Ticket type | **Change**, 73% |
| Sentiment | **Neutral** |
| Summary | "Please update my leave records and let me know which forms I need to submit before my leave begins." *(the greeting "Dear HR team" is skipped)* |
| Recommended action | "Gather error details and reproduction steps, check the knowledge base, and escalate to engineering if unresolved." |

**Expected reply**
> Dear [Customer Name], thank you for your email regarding the update of your parental leave records. We will review your request and provide you with the necessary information to proceed with the update. Please allow us to contact you at [Phone] to discuss the next steps.

**⚠ Known weakness:** the ideal category is *Human Resources*. Human Resources is a small class (2% of test tickets), and the classifier finds only 43% of them, so the action is wrong too. The reply itself is relevant, and the type **Change** is correct.

---

### TC-11: Service outage

**Input**
- Subject: `Website down for all customers`
- Ticket: `Our whole website has been down for the last 30 minutes. Customers cannot place orders and we are losing sales. Is there an outage or maintenance on your side?`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Service Outages and Maintenance**, 45% |
| Priority | **High**, 47% |
| Ticket type | **Incident**, 72% |
| Sentiment | **Negative** |
| Summary | "Our whole website has been down for the last 30 minutes. Customers cannot place orders and we are losing sales." |
| Recommended action | "Handle with high priority. Check the service status dashboard for an ongoing incident and link the ticket to it if one exists." |

**Expected reply**
> We apologize for the inconvenience caused by the downtime of your website. To better assist you, could you please provide details on the error messages you are receiving and the steps you have taken so far? We would like to schedule a call at your convenience to discuss the issue and work towards a resolution. Please let us know a suitable time for the call at [Phone].

**Pass if** the outputs match. Every field is correct here.

---

### TC-12: Ticket without a subject

**Input**
- Subject: *(leave empty)*
- Ticket: `My printer is not connecting to the WiFi network anymore. I restarted the router and the printer but it still does not work.`

**Expected output**

| Field | Expected |
|---|---|
| Category | **Technical Support**, 49% |
| Priority | **High**, 53% |
| Ticket type | **Problem**, 73% |
| Sentiment | **Negative** |
| Summary | "My printer is not connecting to the WiFi network anymore. I restarted the router and the printer but it still does not work." |
| Recommended action | "Handle with high priority. Gather error details and reproduction steps, check the knowledge base, and escalate to engineering if unresolved." |

**Expected reply**
> Hello [Customer Name], we are sorry to hear that your printer is not connecting to the Wi-Fi network. We understand that you have already tried restarting the router and the printer, but the issue still persists. To better assist you, could you please provide more details about the error message you are receiving and the steps you have taken so far? If necessary, we can arrange a call at a convenient time for you to discuss this further. Please let us know a suitable time to contact you at [Phone].

**Pass if** the ticket is accepted without a subject (no error), no subject line appears in the customer card, and the reply mentions what the customer already tried.

---

## 4. Analyze ticket: input validation

### TC-13: Empty ticket
| | |
|---|---|
| **Steps** | Click **Clear**. Leave both fields empty. Click **Analyze ticket**. |
| **Expected** | A red message: **"Enter the customer's ticket text first."** No request is sent and no result appears. |

### TC-14: Only spaces
| | |
|---|---|
| **Steps** | Type a few spaces in **Customer ticket**. Click **Analyze ticket**. |
| **Expected** | The same red message: **"Enter the customer's ticket text first."** |

### TC-15: Too short (under 10 characters)
| | |
|---|---|
| **Steps** | Ticket: `Help me` (7 characters). Click **Analyze ticket**. |
| **Expected** | A red message from the server: **"Ticket: The ticket is too short (minimum 10 characters)."** Nothing is saved, so the ticket count in **Ticket history** does not change. |

### TC-16: Exactly 10 characters (boundary)
| | |
|---|---|
| **Steps** | Ticket: `Need help!` (exactly 10 characters). Click **Analyze ticket**. |
| **Expected** | **Accepted.** A reply and labels appear. The text is meaningless, so check only that it does not error. |

### TC-17: Too long (over 5,000 characters)
| | |
|---|---|
| **Steps** | Paste a long text, for example the TC-07 ticket copied about 30 times, until the counter under the box passes **5,000**. |
| **Expected** | The counter (e.g. `5,190 / 5,000`) turns **red**, and the **Analyze ticket** button becomes **disabled** (greyed out, cannot be clicked). Delete text until it is ≤ 5,000 and the button works again. |

---

## 5. Analyze ticket: buttons

### TC-18: Sample buttons
| | |
|---|---|
| **Steps** | Click each chip above the form: **Refund delay**, **Login / OTP**, **Damaged product**. |
| **Expected** | Each fills both Subject and Customer ticket. They are the same texts as TC-03, TC-04 and TC-05, and analysing them gives the same results as those tests. |

### TC-19: Copy reply
| | |
|---|---|
| **Steps** | After any analysis, click **Copy reply** under the green reply. Paste into a text editor. |
| **Expected** | The button briefly changes to **Copied**. The pasted text is exactly the reply. |

### TC-20: Clear
| | |
|---|---|
| **Steps** | With a ticket and result on screen, click **Clear**. |
| **Expected** | Both fields empty. The right side returns to *"The suggested reply and ticket insights will appear here."* The **Clear** button disappears. |

---

## 6. Ticket history and detail

### TC-21: History lists the analysed tickets
| | |
|---|---|
| **Steps** | Open **Ticket history**. |
| **Expected** | A table with columns **#, Ticket, Category, Priority, Type, Sentiment, Analyzed**. The newest ticket is at the top. Every successful ticket from sections 3–5 is listed (TC-03 to TC-12, TC-16, and any sample-button runs). The TC-12 row shows **(no subject)**. The rejected tickets from TC-13 to TC-15 and TC-17 are **not** listed. |

### TC-22: Ticket detail
| | |
|---|---|
| **Steps** | Click the **Charged twice for my subscription** row. |
| **Expected** | The page **Ticket #N** opens with the same card as in TC-06: the customer ticket, the same reply, Billing and Payments 74%, High 51%, Problem 62%, Negative. |

### TC-23: Pagination
| | |
|---|---|
| **Steps** | Look at the bottom of the history table. |
| **Expected** | Text like **"N tickets, page 1 of X"**. With 20 or fewer tickets, **Previous** and **Next** are both disabled. With more than 20, **Next** shows the older tickets. |

---

## 7. Dashboard

### TC-24: Dashboard numbers
| | |
|---|---|
| **Steps** | Open **Dashboard**. |
| **Expected (if you emptied the database in 1.2 and ran exactly TC-03 to TC-12, 10 tickets)** | See below. |

| Widget | Expected |
|---|---|
| Tickets analyzed | **10** |
| High or critical priority | **60%** (6 of 10) |
| Negative sentiment (inferred) | **70%** (7 of 10) |
| Most common category | **Billing and Payments** |
| Categories chart | Billing and Payments 4 · Technical Support 3 · Customer Service 1 · IT Support 1 · Service Outages and Maintenance 1 |
| Priority chart | High 6 · Medium 4 |
| Ticket types chart | Incident 5 · Problem 3 · Request 1 · Change 1 |
| Sentiment chart | Negative 7 · Neutral 2 · Positive 1 |
| Recent tickets | The 5 newest tickets |
| Response model panel | ROUGE-L **0.158** before fine-tuning, **0.343** after; BERTScore F1 (rescaled) **0.210** before, **0.403** after; test tickets **500** |

If you also ran TC-16 or the sample buttons, the totals are higher. Check that the numbers grew by exactly the number of tickets you added.

### TC-25: Dashboard updates live
| | |
|---|---|
| **Steps** | Note **Tickets analyzed**. Analyze one more ticket (for example TC-11 again). Return to **Dashboard**. |
| **Expected** | **Tickets analyzed** is 1 higher, and the new ticket is first under **Recent tickets**. |

---

## 8. Model comparison

Open **Model comparison**. The form is pre-filled with the refund ticket.

### TC-26: Base vs fine-tuned on the refund ticket
| | |
|---|---|
| **Steps** | Leave the pre-filled text. Click **Compare models**. |
| **Expected** | Three columns. The first run may take longer while the base model loads. |

| Column | Expected |
|---|---|
| **Base FLAN-T5-small (no fine-tuning)** | "Sorry, I haven't received the refund yet. I'm sorry, but I've been unable to get the refund." It writes *as if it were the customer*, not a support agent. |
| **Fine-tuned FLAN-T5-small (ours)** (highlighted) | The same reply as TC-03 ("Dear [Customer Name], thank you for contacting us regarding the refund you received…") |
| **Gemini (optional benchmark)** | **Not set up**: "Set GEMINI_API_KEY in .env to enable the Gemini benchmark." |

**Pass if** the base reply is clearly not a support reply and the fine-tuned one is. This is the key evidence that fine-tuning worked.

### TC-27: Comparison on the duplicate-charge ticket
| | |
|---|---|
| **Steps** | Replace the text with the TC-06 subject and ticket. Click **Compare models**. |
| **Expected** | **Base:** "Your subscription has been charged twice for your monthly subscription. Please refund the duplicate payment as soon as possible." It only repeats the customer's words. **Fine-tuned:** the full reply from TC-06. |

### TC-28: Comparison on the server-crash ticket
| | |
|---|---|
| **Steps** | Replace the text with the TC-07 subject and ticket. Click **Compare models**. |
| **Expected** | **Base:** "Sorry, I can't help you with this problem. Please help me fix it." Unhelpful and confused. **Fine-tuned:** the full reply from TC-07. |

### TC-29: Offline benchmark table and validation
| | |
|---|---|
| **Steps** | 1. Scroll down to **Offline benchmark on 100 test tickets**. 2. Then clear the **Customer ticket** box and click **Compare models**. |
| **Expected** | 1. Table values for base → fine-tuned: ROUGE-1 **0.223 → 0.483**, ROUGE-L **0.177 → 0.351**, BLEU **1.28 → 20.15**, BERTScore F1 (rescaled) **0.225 → 0.408**, semantic cosine **0.613 → 0.787**, average words per reply **19.4 → 66.0**. 2. A red error: **"Ticket: The customer ticket cannot be empty."** |

---

## 9. Model evaluation

### TC-30: Evaluation page shows the Colab results
| | |
|---|---|
| **Steps** | Open **Model evaluation** and scroll through every section. |
| **Expected** | Sections: **Fine-tuning**, **Response quality: before vs after fine-tuning**, **AI-judge evaluation (LLM-as-a-judge)**, **Supporting classifiers**, **Dataset preparation**, **Example replies**, **Limitations**. There is **no** significance table, **no** "Size and speed" section and **no** confusion-matrix images. Key values: ROUGE-L **0.158 → 0.343**, BLEU **1.17 → 19.57**, BERTScore F1 **0.867 → 0.899**, judge overall **1.59 → 4.03** with the fine-tuned model preferred on **50 of 50**, classifier accuracy type **0.881**, priority **0.643**, queue **0.620**, split **18,848 / 2,354 / 2,339**. The Epochs row reads **4 (all epochs run, validation loss checked after each)**. |

### TC-30b: Macro-F1 explanation panel
| | |
|---|---|
| **Steps** | On **Model evaluation**, scroll to **Supporting classifiers** and the panel below the table. |
| **Expected** | Title **"Is a macro-F1 of 0.59 too low?"** with the short answer "it is moderate, not bad". A bar chart with three bars per target: **Always guess the most common class**, **Random guessing** and **Our classifier**, where our bar is clearly highest for Category, Priority and Ticket type. Five numbered reasons, the first naming **General Inquiry (45 test tickets)** and **Technical Support (653)**, with plain accuracy **62%**. Two yellow boxes: **How the app handles it** and **How to improve it**. |

### TC-30c: How it works page
| | |
|---|---|
| **Steps** | Click **How it works** in the sidebar (second item). Scroll to the bottom. Click the **Analyze ticket** link in step 6. |
| **Expected** | Title **How SupportGen works**, a problem/idea box, then 8 numbered steps: **The big picture** (4 boxes with arrows: Data → Training → Backend → Website), **The data** (61,765 raw → 28,261 English → 23,557 after cleaning; split 18,848 / 2,354 / 2,339), **Training the model** (18,848 examples, 30 minutes on a Tesla T4, 4 epochs, with a before/after reply for the server-crash ticket), **What happens when you click "Analyze ticket"** (a 5-row table), **How we know it works** (0.16 → 0.34, 0.58 → 0.78, 1.6 → 4.0, 50 / 50), **The website** (links to the pages), **Limits**, **Built with**. The link opens the Analyze page. |

---

## 10. Error handling

### TC-31: Backend stopped
| | |
|---|---|
| **Steps** | 1. Press `Ctrl + C` in Terminal 1 to stop Django. 2. Reload http://localhost:5173. 3. Open **Analyze ticket** and analyse any valid ticket. |
| **Expected** | The sidebar shows **Backend offline** (red dot). Analyzing shows a red error message, e.g. "Cannot reach the SupportGen server. Start the Django backend on port 8000." or "Request failed (HTTP 500)." The page does not crash or go blank. |
| **After** | Start the backend again (`python manage.py runserver`) and reload; the sidebar is green again. |

### TC-32: Unknown page
| | |
|---|---|
| **Steps** | Open http://localhost:5173/does-not-exist |
| **Expected** | The sidebar still shows, and the main area says **Page not found.** |

---

## 11. Results log

Copy this table and fill it in while testing.

| Test | Title | Result (Pass / Fail) | Notes |
|---|---|---|---|
| TC-01 | Sidebar status | | |
| TC-02 | Health endpoint | | |
| TC-03 | Refund not received ⚠ | | |
| TC-04 | Login / OTP | | |
| TC-05 | Cracked laptop ⚠ | | |
| TC-06 | Charged twice | | |
| TC-07 | Server crashing | | |
| TC-08 | Enterprise pricing ⚠ | | |
| TC-09 | Positive feedback ⚠ | | |
| TC-10 | HR request ⚠ | | |
| TC-11 | Website outage | | |
| TC-12 | No subject | | |
| TC-13 | Empty ticket | | |
| TC-14 | Only spaces | | |
| TC-15 | Too short | | |
| TC-16 | Exactly 10 characters | | |
| TC-17 | Too long | | |
| TC-18 | Sample buttons | | |
| TC-19 | Copy reply | | |
| TC-20 | Clear | | |
| TC-21 | History list | | |
| TC-22 | Ticket detail | | |
| TC-23 | Pagination | | |
| TC-24 | Dashboard numbers | | |
| TC-25 | Dashboard updates | | |
| TC-26 | Compare: refund | | |
| TC-27 | Compare: duplicate charge | | |
| TC-28 | Compare: server crash | | |
| TC-29 | Benchmark table + validation | | |
| TC-30 | Evaluation page | | |
| TC-30b | Macro-F1 explanation | | |
| TC-30c | How it works page | | |
| TC-31 | Backend stopped | | |
| TC-32 | Unknown page | | |

**Summary of what the ⚠ cases show:** the reply model can mishandle negation (TC-03), invent facts (TC-08) and treat positive feedback as a complaint (TC-09). The queue classifier confuses topics with overlapping words (TC-05, TC-08, TC-10). All of these match the limitations in the README and are why every reply is shown as a draft for a human to review.

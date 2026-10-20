"""Out-of-scope texts for Laya, hand-written (fictional). Used in the notebook (part 5) to check the detector in
layamed/scope.py: non-medical texts should be blocked; medical cases of other diseases show the detector's limit
(it separates "clinical case" from "any text", not "one of the 4 diseases" from "another disease").
"""
NON_MEDICAL = [
    "Yesterday I went to the supermarket with my sister to buy groceries for the weekend. We picked up bread, eggs, "
    "milk, a few apples and some cheese for the kids. The store was crowded because of the holiday sale, so we waited "
    "almost twenty minutes in line. On the way back home it started raining heavily and the traffic was terrible. "
    "When we finally arrived, my dog was waiting at the door, very excited to see us.",
    "The quarterly report shows that revenue increased by twelve percent compared with the same period last year, "
    "driven mainly by strong sales in the northern region. Operating costs remained stable, although marketing "
    "expenses rose after the launch of the new product line. The board expects margins to improve next quarter as "
    "the supply chain issues are resolved. Shareholders will receive the full statement at the annual meeting in March.",
    "To make the tomato sauce, heat two tablespoons of olive oil in a large pan over medium heat. Add one chopped "
    "onion and three cloves of garlic and cook for five minutes until soft. Pour in the crushed tomatoes, add salt, "
    "pepper and a pinch of sugar, then simmer for thirty minutes, stirring occasionally. Finish with fresh basil "
    "leaves and serve over pasta with grated parmesan cheese.",
    "The home team scored twice in the second half and won the championship final in front of a sold-out stadium. "
    "The striker, who had missed a penalty in the first half, made up for it with a powerful header in the 78th "
    "minute. The coach praised the defense for resisting the pressure during the last ten minutes. Fans celebrated "
    "in the streets until late at night, and the parade is planned for Sunday morning.",
    "This function reads the configuration file, validates each field and returns a dictionary with the parsed "
    "values. If a required key is missing, it raises a ValueError with the name of the key. Optional keys receive "
    "default values defined at the top of the module. The parser ignores comments and blank lines, and it supports "
    "nested sections separated by dots. Unit tests cover the main error cases and the default behavior.",
    "Dear Mr. Johnson, thank you for your email regarding the delivery of the office furniture. Unfortunately the "
    "shipment was delayed at the port because of customs inspections. We now expect the desks and chairs to arrive "
    "next Thursday, and our team will call you to confirm the installation time. We apologize for the inconvenience "
    "and will offer a ten percent discount on your next order. Kind regards, Sarah from customer service.",
    "The Roman Empire reached its greatest territorial extent under the emperor Trajan in the second century. Its "
    "roads, aqueducts and legal system influenced Europe for centuries after the fall of the western half in 476. "
    "Historians still debate the causes of the decline, pointing to economic crisis, political instability, military "
    "pressure on the borders and the division of the empire into eastern and western administrations.",
    "Our hotel was located just two blocks from the beach, with a beautiful view of the bay from the balcony. "
    "Breakfast was included and offered fresh fruit, pastries and excellent coffee. During the week we visited the "
    "old town, took a boat tour to the islands and tried the local seafood at a small restaurant near the harbor. "
    "The staff were friendly and helpful, and we would definitely come back next summer.",
    "The tenant agrees to pay the monthly rent on or before the fifth day of each month. Late payments will incur a "
    "fee of fifty dollars. The landlord is responsible for major repairs to the roof, plumbing and electrical system, "
    "while the tenant must keep the apartment clean and report any damage within seven days. Either party may end "
    "this agreement with sixty days of written notice before the end of the lease term.",
    "I bought this vacuum cleaner three months ago and I am very happy with it. It is light, quiet and the battery "
    "lasts about forty minutes, which is enough for the whole apartment. The brush works well on carpets and picks up "
    "pet hair easily. The only negative point is that the dust container is small and needs to be emptied often. "
    "Overall I would recommend it to anyone looking for a cordless model at a reasonable price.",
    "The new smartphone features a larger display, a faster processor and an improved camera system with three "
    "lenses. Battery life has been extended by about two hours compared with the previous model. The device supports "
    "wireless charging and comes in four colors. Pre-orders open on Friday, and the phone will be available in "
    "stores at the end of the month, starting at seven hundred and ninety-nine dollars.",
    "Climate scientists warn that global average temperatures could rise by more than two degrees this century if "
    "emissions are not reduced. Melting glaciers and rising sea levels already threaten coastal cities, and extreme "
    "weather events such as droughts, floods and heat waves are becoming more frequent. Governments met last month "
    "to discuss new targets, but environmental groups say the commitments are still not enough.",
    "My grandfather was born in a small village in the mountains, where his family grew corn and raised goats. When "
    "he was eighteen he moved to the city to work in a textile factory. He met my grandmother at a dance, and they "
    "married two years later. He loved telling stories about the old days, especially about the winter when the "
    "snow blocked the only road to the village for almost two months.",
    "The meeting is scheduled for Tuesday at 10 a.m. in conference room B. Please review the attached slides before "
    "the session and bring your comments on the budget proposal. We will discuss the hiring plan for the next "
    "quarter, the timeline for the website redesign and the results of the customer satisfaction survey. Lunch will "
    "be provided. If you cannot attend, please send your feedback to the project manager by Monday afternoon.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
    "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est.",
    "To change a flat tire, first park the car on a flat surface and turn on the hazard lights. Loosen the lug nuts "
    "slightly before lifting the vehicle with the jack. Remove the nuts and the damaged tire, then mount the spare "
    "tire on the wheel bolts. Tighten the nuts by hand, lower the car and finish tightening them in a star pattern. "
    "Check the pressure of the spare as soon as possible at a gas station.",
    "The museum's new exhibition brings together more than two hundred paintings from the impressionist period, many "
    "of them on loan from private collections. Visitors can follow the development of the movement from the first "
    "outdoor landscapes to the late works on light and color. Tickets must be booked online, and guided tours are "
    "offered in three languages every afternoon. The exhibition runs until the end of January.",
    "The city council approved the plan to build a new bike lane along the main avenue, connecting the train station "
    "to the university campus. Construction will start in the spring and is expected to take eight months. Some "
    "shop owners worry about the loss of parking spaces, while residents say the project will reduce traffic and "
    "make the streets safer for children. A public hearing will be held next week.",
    "Ontem fui ao mercado com a minha mãe e compramos frutas, verduras e pão para o café da manhã. Depois passamos na "
    "farmácia e na padaria, e voltamos para casa a pé porque o dia estava bonito. À tarde assisti a um filme com os "
    "meus amigos e à noite jantamos pizza. Amanhã pretendo acordar cedo para estudar para a prova de matemática e "
    "depois ir à academia.",
    "asdf qwer zxcv tyui ghjk bnm lkjh poiu mnbv asdf qwer zxcv tyui ghjk bnm lkjh poiu mnbv asdf qwer zxcv tyui "
    "ghjk bnm lkjh poiu mnbv asdf qwer zxcv tyui ghjk bnm lkjh poiu mnbv asdf qwer zxcv tyui ghjk bnm lkjh poiu "
    "mnbv asdf qwer zxcv tyui ghjk bnm lkjh poiu mnbv asdf qwer zxcv tyui ghjk bnm lkjh poiu mnbv asdf qwer zxcv",
    "test test test test test test test test test test test test test test test test test test test test test test "
    "test test test test test test test test test test test test test test test test test test test test test test "
    "test test test test test test test test test test test test test test test test test test test test test test "
    "test test test test test test test test test test test test test test test test test test test test test test",
    "Once upon a time, in a kingdom far away, there lived a young princess who loved to explore the forest behind the "
    "castle. One morning she found a small wounded fox near the river and decided to take care of it. Every day she "
    "brought it food and water, and slowly the fox recovered. When the king discovered the secret, he was angry at "
    "first, but later he allowed the fox to live in the royal garden.",
    "The train to the airport leaves every fifteen minutes from platform four. The journey takes about thirty-five "
    "minutes and tickets can be bought at the machines or with the mobile app. On weekends the service starts at six "
    "in the morning and ends at midnight. Passengers with large suitcases should use the last car, which has more "
    "space for luggage. Delays are announced on the screens in the main hall.",
    "Our startup is building a platform that helps small farmers sell their products directly to restaurants. The "
    "app connects buyers and sellers, handles payments and organizes delivery routes. In the first pilot, forty "
    "farms and twenty restaurants used the platform for three months, and sales grew steadily every week. We are now "
    "raising a seed round to expand to two more cities and hire a small engineering team.",
    "The garden needs to be watered early in the morning during the summer, when the soil is still cool. Tomatoes "
    "and peppers should be planted in full sun, while lettuce prefers partial shade. Add compost to the beds every "
    "spring and remove weeds regularly to avoid competition for nutrients. Prune the rose bushes in late winter and "
    "protect young plants from snails with a ring of crushed eggshells.",
    "Students must submit the final project by the end of the semester. The project should include a written report "
    "of no more than ten pages, the source code and a short presentation of the results. Groups of up to three "
    "students are allowed. Late submissions will lose ten percent of the grade per day. Office hours are on "
    "Wednesdays from two to four in the afternoon, and questions can also be posted on the course forum.",
    "The storm caused power outages in several neighborhoods and knocked down trees along the coastal road. "
    "Firefighters worked through the night to clear the streets and rescue families from flooded houses. Schools "
    "will remain closed tomorrow, and the authorities recommend that residents avoid unnecessary travel. The weather "
    "service expects the rain to stop by Thursday, followed by a period of cooler temperatures.",
    "I have been learning to play the guitar for six months. At first my fingers hurt a lot and I could barely play "
    "a single chord, but now I can play several songs from beginning to end. I practice for about thirty minutes "
    "every evening after dinner. My teacher says I should spend more time on rhythm and less on learning new songs. "
    "Next month I want to try playing with a small group of friends.",
    "Bitcoin fell sharply on Monday after regulators announced new rules for cryptocurrency exchanges. Analysts said "
    "investors were worried about stricter reporting requirements and possible limits on trading. Other digital "
    "currencies followed the same trend, losing between five and ten percent of their value. Some experts believe "
    "the market will recover in the coming weeks, while others expect more volatility until the rules are clear.",
    "Please assemble the bookshelf following the steps in this manual. First attach the side panels to the bottom "
    "board using the long screws. Then insert the shelves into the pre-drilled holes and secure them with the small "
    "metal pins. Fix the back panel with the nails provided and attach the safety strap to the wall. Do not place "
    "more than twenty kilograms on each shelf. Keep the instructions for future reference.",
]

# To reach the interface's minimum context, each long text joins two consecutive texts of the list
NON_MEDICAL_LONG = [a + " " + b for a, b in zip(NON_MEDICAL, NON_MEDICAL[1:] + NON_MEDICAL[:1])]

# 8 more non-medical texts, written after the method was chosen (physics, job ad, code, JSON, prose, numbers,
# Portuguese, marathon): never used to choose anything, only to test
NON_MEDICAL_NEW = [
    "Quantum entanglement is a physical phenomenon that occurs when a group of particles is generated, interacts, or shares spatial proximity in a way such that the quantum state of each particle cannot be described independently of the state of the others. Measurements of physical properties such as position, momentum, spin, and polarization performed on entangled particles can be found to be perfectly correlated. Einstein famously called it spooky action at a distance, and later experiments confirmed the violation of Bell inequalities in many laboratories around the world.",
    "We are looking for a senior data engineer to join our analytics team. You will design and maintain data pipelines, work closely with product managers and data scientists, and ensure the quality and reliability of our reporting systems. Requirements: five years of experience with Python and SQL, familiarity with cloud platforms, and strong communication skills. We offer flexible hours, remote work two days a week, health insurance and a yearly training budget. Send your resume and a short cover letter to our recruiting team.",
    "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    middle = len(items) // 2\n    left = merge_sort(items[:middle])\n    right = merge_sort(items[middle:])\n    result = []\n    while left and right:\n        result.append(left.pop(0) if left[0] <= right[0] else right.pop(0))\n    return result + left + right\n\nprint(merge_sort([5, 3, 9, 1, 4, 8, 2, 7, 6]))\n# This function sorts a list using the divide and conquer approach, splitting the list and merging the sorted halves.",
    '{"order_id": 48213, "customer": {"name": "Anna", "city": "Lisbon"}, "items": [{"sku": "A-100", "qty": 2, "price": 19.9}, {"sku": "B-220", "qty": 1, "price": 49.0}], "shipping": "express", "status": "delivered", "notes": "leave the package at the reception desk", "created_at": "2026-03-14T10:22:00Z", "updated_at": "2026-03-16T08:01:00Z"}',
    "The autumn leaves fall slowly on the quiet road, golden and red under a pale sky. The old man walks with his dog every morning, past the bakery and the closed bookstore, thinking about the summers of his youth by the sea. Children run to school with heavy backpacks, laughing and shouting. In the evening the streetlights come on one by one, and the smell of chestnuts fills the square near the church, where the musicians play until late.",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80",
    "O Brasil é o maior país da América do Sul e tem uma enorme diversidade cultural e natural. A floresta amazônica ocupa grande parte do território do norte, enquanto o sul tem clima mais frio e paisagens de campos. A culinária varia muito de região para região, com pratos como feijoada, acarajé, churrasco e pão de queijo. O futebol é uma paixão nacional, e o carnaval atrai milhões de turistas todos os anos para cidades como Rio de Janeiro e Salvador.",
    "The marathon starts at 7 a.m. near the city hall. Runners should arrive one hour early to collect their numbers and leave their bags at the storage tent. Water stations are placed every five kilometers, and energy gels will be offered at kilometers 20 and 30. The course closes after six hours. Volunteers will guide participants at every intersection, and the medal ceremony takes place in the central park after the last runner crosses the finish line.",
]
NON_MEDICAL_NEW_LONG = [a + " " + b for a, b in zip(NON_MEDICAL_NEW, NON_MEDICAL_NEW[1:] + NON_MEDICAL_NEW[:1])]

# Medical, but outside the 4 diseases (fictional patients): history, physical exam and labs
OTHER_DISEASES = [
    ("68-year-old man with hypertension and smoking history presents with crushing substernal chest pain for 2 hours, "
     "radiating to the left arm and jaw, associated with diaphoresis and shortness of breath. No abdominal pain. "
     "Pain started at rest and did not improve with rest. He had similar but shorter episodes on exertion last week.",
     "T 36.9 C, HR 104, BP 158/94, RR 22, SpO2 94%. Diaphoretic, anxious. Heart regular, no murmurs. Bibasilar "
     "crackles. Abdomen soft, non-tender. No leg edema.",
     "Troponin T: 0.85 ng/mL, Creatine Kinase (CK): 420.0 IU/L, White Blood Cells: 10.2 K/uL, Hemoglobin: 14.1 g/dL, "
     "Creatinine: 1.0 mg/dL, Potassium: 4.2 mEq/L"),
    ("34-year-old woman presents with 4 days of productive cough, fever to 39 C, pleuritic right-sided chest pain and "
     "shortness of breath. She reports chills and fatigue. No abdominal pain, nausea or vomiting. No recent travel. "
     "She smokes half a pack per day. No history of asthma or lung disease.",
     "T 39.1 C, HR 112, BP 112/70, RR 26, SpO2 91% on room air. Dullness to percussion and bronchial breath sounds "
     "over the right lower lung field. Abdomen soft, non-tender.",
     "White Blood Cells: 17.8 K/uL, Neutrophils: 88.0 %, C-Reactive Protein: 186.0 mg/L, Lactate: 2.1 mmol/L, "
     "Sodium: 133.0 mEq/L, Creatinine: 0.9 mg/dL"),
    ("75-year-old woman brought by family for sudden onset of right-sided weakness and difficulty speaking that began "
     "1 hour ago while having breakfast. She has atrial fibrillation and stopped her anticoagulant last month. No "
     "headache, no seizure, no trauma. No abdominal complaints.",
     "T 36.7 C, HR 96 irregular, BP 186/98. Expressive aphasia, right facial droop, right arm and leg strength 2/5. "
     "Abdomen soft, non-tender.",
     "Glucose: 128.0 mg/dL, INR(PT): 1.1, Platelet Count: 245.0 K/uL, White Blood Cells: 8.4 K/uL, "
     "Creatinine: 0.9 mg/dL, Sodium: 139.0 mEq/L"),
    ("25-year-old woman with 2 days of burning on urination, increased frequency and urgency, and suprapubic "
     "discomfort. No fever, no flank pain, no vaginal discharge. Sexually active, last menstrual period one week ago. "
     "No nausea or vomiting. No prior urinary infections.",
     "T 37.2 C, HR 84, BP 116/72. Mild suprapubic tenderness, abdomen otherwise soft. No costovertebral angle "
     "tenderness. No rebound or guarding.",
     "White Blood Cells: 9.1 K/uL, Urine Leukocytes: Large, Urine Nitrite: Positive, Urine Blood: Small, "
     "Creatinine: 0.7 mg/dL, hCG: Negative"),
]

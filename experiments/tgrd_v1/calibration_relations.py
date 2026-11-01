"""Reviewed relations for the 34 accepted cal40 teachers, CPU only.

The frozen ten-root bootstrap remains unchanged. These are explicit authored
templates, not a general relation parser, independent human review, or evidence
that all 7473 training roots have acceptable teachers/probes.
"""
from __future__ import annotations

from pathlib import Path
from bootstrap_relations import spec, templates, specifications as bootstrap_specs
from runtime_full import ROOT, HERE, DATA, OUT, read, freeze, save, checked_rows, baseline
from variants import build

RELATIONS = DATA/'cal40_reviewed_relations_v2.json'
VARIANTS = DATA/'cal40_reviewed_variants_v2.json'
MANIFEST = DATA/'cal40_reviewed_manifest_v2.json'
REVIEWS = OUT/'material_audits/cal40_teacher_full_text_reviews_v1.json'


def additional_specifications():
    result = []

    def add(index, params, domains, derived, original, numeric, wording, structural,
            changed, reason, *, integer=True, nonnegative=(), integers=()):
        value = spec(index, params, domains, derived, original,
            templates(f'cal40_{index}', numeric, wording, structural, original, changed, integer),
            reason, nonnegative)
        value['integer_constraints'] = list(integers)
        value['sampling_parameter_order'] = list(domains)
        result.append(value)

    add(5753, {'total':55,'boys':30,'long_num':3,'denom':5},
        {'total':[40,50,60,70,80],'boys':[10,15,20,25,30],
         'long_num':[1,2,3,4],'denom':[5,6,8,10]},
        {'girls':'total-boys','short':'girls*(denom-long_num)/denom'}, 'short',
        ['A track meet has {total} people, including {boys} boys. Of the girls, {long_num}/{denom} have long hair and the rest have short hair. How many girls have short hair?',
         'There are {total} entrants at a meet, of whom {boys} are boys. The remaining entrants are girls; {long_num}/{denom} of the girls have long hair. How many girls have short hair?'],
        ['Among {total} people, {boys} are boys and the others are girls. Girls with long hair make up {long_num}/{denom} of all girls. How many girls are in the short-haired group?',
         'A group contains {total} children, {boys} of them boys. The other children are girls. The long-haired share of the girls is {long_num}/{denom}. All other girls have short hair. How many girls have short hair?'],
        ['At a meet there are {boys} boys and some girls. Of the girls, {long_num}/{denom} have long hair; the other {short} girls have short hair. How many people are at the meet altogether?',
         'A group has {boys} boys. The girls are divided into a long-haired fraction {long_num}/{denom} and {short} short-haired girls. What is the total number of people in the group?'],
        'boys+short*denom/(denom-long_num)',
        'Remove boys before taking the complementary fraction of girls: (55-30)*(1-3/5)=10. The inverse recovers girls from the short-haired subgroup and adds boys.',
        nonnegative=['girls','denom-long_num-1'], integers=['short','girls-short'])

    add(4793, {'a':1,'factor':2,'b':2,'extra':1,'days':4},
        {'a':[1,2,3,4],'factor':[2,3,4,5],'b':[2,3,4,5],
         'extra':[1,2,3,4],'days':[2,3,4,5]},
        {'rate':'a+a*factor+b+b+extra','growth':'days*(a+a*factor+b+b+extra)'}, 'growth',
        ['Four trees grow steadily. The first grows {a} meters per day. The second grows the same length in 1/{factor} of the time the first needs. The third grows {b} meters per day, and the fourth grows {extra} meters per day more than the third. How many meters do all four grow together in {days} days?',
         'Tree A grows {a} meters each day. Tree B takes 1/{factor} as long as A to grow an equal length. Tree C grows {b} meters per day and tree D grows {extra} meters more per day than C. What is their combined growth over {days} days?'],
        ['There are four constant growth rates: {a} meters daily for the first tree; the same growth in 1/{factor} of the time for the second; {b} meters daily for the third; and {extra} additional meters daily beyond the third for the fourth. How many meters of combined growth occur in {days} days?',
         'A gardener measures four trees for {days} days. The first grows {a} meters per day; the second needs only 1/{factor} of the first tree\'s time for an equal gain; the third gains {b} meters daily; and the fourth gains {extra} meters more daily than the third. What is the sum of their growth in meters?'],
        ['Four trees together have grown {growth} meters. Their constant daily rates are {a}, {a} times {factor}, {b}, and {b} plus {extra} meters. For how many days have they grown?',
         'The first tree grows {a} meters daily, the second {factor} times that rate, the third {b} meters daily, and the fourth {extra} meters more daily than the third. How many days produce {growth} meters of combined growth?'],
        'growth/rate',
        'Equal growth in half the time doubles the rate. Original rates 1,2,2,3 sum to8 and four days give32; the inverse divides combined growth by the summed rate.')

    add(2386, {'intake':12,'denom':2,'adopted':3,'kittens':5,'returned':1},
        {'intake':[12,24,36,48],'denom':[2,3,4,6],'adopted':[1,2,3,4],
         'kittens':[2,3,4,5],'returned':[1,2,3,4]},
        {'before':'intake/denom','after_adoption':'intake+intake/denom-adopted',
         'final':'intake+intake/denom-adopted+kittens-returned'}, 'final',
        ['A shelter takes in {intake} cats and already has 1/{denom} as many cats as this intake. Then {adopted} cats are adopted, a remaining cat has {kittens} kittens, and {returned} cats are collected by their owners. How many cats are in the shelter now?',
         'An animal shelter initially has 1/{denom} of {intake} cats. It receives {intake} additional cats. After {adopted} adoptions, one cat gives birth to {kittens} kittens, and owners take home {returned} cats. What is the final cat count?'],
        ['A shelter adds {intake} cats to an existing population equal to 1/{denom} of that intake. It loses {adopted} cats to adoption, gains {kittens} newborn kittens from one remaining mother, then returns {returned} cats to owners. How many cats remain?',
         'Before receiving {intake} cats, a shelter has one cat for every {denom} in the new intake. Next {adopted} cats leave for adoption; a cat still there has {kittens} kittens; finally {returned} cats leave with owners. How many cats are there at the end?'],
        ['A shelter receives {intake} cats in addition to an existing population of 1/{denom} of that intake. It adopts out {adopted} cats; a remaining cat has some kittens; and owners collect {returned} cats. There are {final} cats at the end. How many kittens were born?',
         'A shelter starts with 1/{denom} of {intake} cats, receives {intake}, and places {adopted} for adoption. After a litter is born and {returned} cats are returned to owners, {final} cats remain. How many kittens were in the litter?'],
        'final-intake-intake/denom+adopted+returned',
        'The initial half refers to the12-cat intake, not a changing stock. Original6+12-3+5-1=19. The mother is already counted; inverse solves the birth increment.',
        nonnegative=['after_adoption-1','after_adoption-returned','final'], integers=['before'])

    add(607, {'total':1000000,'denom':2,'siblings':4},
        {'total':[12000,24000,48000,96000],'denom':[2,3,4,5],
         'siblings':[4,5,6,8]},
        {'keep':'total/denom','share':'total*(denom-1)/denom/siblings',
         'difference':'total/denom-total*(denom-1)/denom/siblings'}, 'difference',
        ['Sandy has {total} tokens. She keeps 1/{denom} of them and divides all remaining tokens equally among {siblings} siblings. How many more tokens does Sandy have than one sibling?',
         'A person keeps 1/{denom} of {total} tokens and shares the rest equally with {siblings} siblings. How many tokens greater is her own holding than each sibling\'s holding?'],
        ['Of {total} tokens, Sandy retains one of {denom} equal parts. She divides the other parts equally among {siblings} siblings. What is the difference between Sandy\'s tokens and one sibling\'s tokens?',
         'Sandy owns {total} tokens before sharing. Her personal fraction is 1/{denom}; the rest goes evenly to {siblings} siblings. By how many tokens does her allocation exceed the allocation to each sibling?'],
        ['Sandy keeps 1/{denom} of a stock of tokens and divides the rest equally among {siblings} siblings. She then has {difference} more tokens than each sibling. How many tokens were in the stock?',
         'A person retains 1/{denom} of her tokens and splits the remainder among {siblings} siblings equally. Her holding exceeds one sibling\'s by {difference} tokens. What was her original number of tokens?'],
        'difference/(1/denom-(denom-1)/denom/siblings)',
        'Subtract one sibling\'s equal remainder share, not the whole remainder. Original500000-125000=375000. Inverse coefficient is positive; all allocations are integer.',
        nonnegative=['difference-1'], integers=['keep','share'])

    add(5574, {'weeks':3,'weekday':3,'saturday':5},
        {'weeks':[2,3,4,5],'weekday':[1,2,3,4],'saturday':[2,3,4,5]},
        {'hours':'weeks*(5*weekday+saturday)'}, 'hours',
        ['Mike practices {weekday} hours on each weekday and {saturday} hours every Saturday, resting on Sunday. How many hours does he practice in {weeks} complete weeks?',
         'A goalkeeper trains {weekday} hours per weekday and {saturday} hours each Saturday, with no Sunday training. What is the total training time in {weeks} full weeks?'],
        ['Each week contains five training days of {weekday} hours apiece plus a Saturday session of {saturday} hours. Sunday is a rest day. How many training hours accumulate over {weeks} weeks?',
         'For {weeks} entire weeks, a player trains {weekday} hours on Monday through Friday, {saturday} hours on Saturday, and zero hours on Sunday. How many hours does the player train?'],
        ['Mike practices {weekday} hours each weekday and {saturday} hours every Saturday, never on Sundays. After how many complete weeks does he accumulate {hours} hours?',
         'With five daily sessions of {weekday} hours and one Saturday session of {saturday} hours per week, how many full weeks yield {hours} hours of practice?'],
        'hours/(5*weekday+saturday)',
        'There are five weekdays, not seven paid training days. Original3*(5*3+5)=60. Whole-week variants avoid partial-week boundary ambiguity.')

    add(1075, {'hoodie':80,'flash_percent':20,'boots':110,'discount':10},
        {'hoodie':[40,60,80,100],'flash_percent':[10,20,25,50],
         'boots':[80,100,120,140],'discount':[10,20,25,30]},
        {'total':'hoodie*(1+flash_percent/100)+boots*(1-discount/100)'}, 'total',
        ['A hoodie costs ${hoodie}. A flashlight costs {flash_percent} percent of the hoodie price. Boots listed at ${boots} have a {discount} percent discount. How many dollars do these three purchases cost in total?',
         'Celina buys a ${hoodie} hoodie, a flashlight priced at {flash_percent} percent of that hoodie, and boots whose ${boots} list price is reduced by {discount} percent. What is her total spending in dollars?'],
        ['A hiker pays full price for a hoodie costing ${hoodie}, {flash_percent} percent of the hoodie price for a flashlight, and 100 percent minus {discount} percent of a ${boots} list price for boots. How many dollars does she spend?',
         'The hoodie is ${hoodie}. The flashlight price is {flash_percent} percent of this amount. The boots are {discount} percent cheaper than their original ${boots} price. What is the total cost of the hoodie, flashlight, and boots?'],
        ['A hoodie costs ${hoodie}, its flashlight costs {flash_percent} percent as much, and boots are discounted {discount} percent. The combined spending is ${total}. What was the undiscounted price of the boots in dollars?',
         'A hiker spends ${total} on a ${hoodie} hoodie, a flashlight costing {flash_percent} percent of the hoodie price, and discounted boots. The boots were reduced by {discount} percent. What was their original price in dollars?'],
        '(total-hoodie*(1+flash_percent/100))/(1-discount/100)',
        'Flashlight percent uses the hoodie base; boot discount uses the boot base. Original80+16+99=195. The inverse removes hoodie/flashlight costs before undoing the boot discount.',
        integer=False)

    add(7014, {'initial':50,'first':1,'floated':12,'later':9,'received':11},
        {'initial':[50,60,70,80],'first':[1,2,3,4],'floated':[5,10,12,15],
         'later':[5,7,9,11],'received':[6,8,10,12]},
        {'final':'initial-first-floated-later+received'}, 'final',
        ['Claire starts with {initial} balloons. She gives away {first}, loses {floated} when they float away, gives away {later} more, and receives {received} from a coworker. How many balloons does she have now?',
         'A worker has {initial} balloons. After giving {first} to children, {floated} balloons escape. She gives out another {later} and takes {received} from a colleague. What is her final balloon count?'],
        ['From a stock of {initial} balloons, Claire distributes {first}, loses {floated}, and distributes another {later}. A colleague then adds {received} to her stock. How many balloons remain with Claire?',
         'Claire\'s balloons change by three separate removals: {first} handed out, {floated} floating off, and {later} handed out later. Starting with {initial}, she finally receives {received} more. How many does she hold at the end?'],
        ['Claire starts with {initial} balloons, gives away {first}, loses {floated}, and gives away another {later}. After receiving some from a coworker she has {final}. How many did the coworker give her?',
         'A worker begins with {initial} balloons and removes {first}, {floated}, and {later} in three separate events. A colleague adds enough to leave her with {final}. How many balloons were added?'],
        'final-initial+first+floated+later',
        'Gifts and escaped balloons are separate losses; coworker transfer is an addition. Original50-1-12-9+11=39. Inverse solves the added transfer.',
        nonnegative=['initial-first-floated-later'])

    add(4488, {'distance':60,'fast':30,'slow':12},
        {'distance':[24,36,48,60],'fast':[24,30,36,48],'slow':[6,8,10,12]},
        {'wait':'distance/slow-distance/fast'}, 'wait',
        ['Two boats start together and cross a lake {distance} miles wide at constant speeds of {fast} and {slow} miles per hour. How many hours does the faster boat wait at the far shore for the slower one?',
         'A speedboat at {fast} miles per hour and a sailboat at {slow} miles per hour leave the same shore at the same time. Each travels {distance} miles. What is the wait in hours between their arrivals?'],
        ['Over the same {distance}-mile crossing, two boats travel steadily at {fast} and {slow} miles per hour from a simultaneous start. How many hours separate their finish times?',
         'Sisters depart together on a {distance}-mile lake journey. One boat maintains {fast} miles per hour and the other {slow} miles per hour. How long in hours must the earlier arrival wait?'],
        ['Two boats start simultaneously across a lake at {fast} and {slow} miles per hour. The faster boat arrives {wait} hours earlier. What is the lake crossing distance in miles?',
         'At constant speeds of {fast} and {slow} miles per hour, boats covering the same distance from a joint start finish {wait} hours apart. How many miles long is the crossing?'],
        'wait/(1/slow-1/fast)',
        'Waiting time is slow travel time minus fast travel time, not distance divided by a speed difference. Original60/12-60/30=3. Simultaneous start is explicit in probes.',
        integer=False, nonnegative=['fast-slow-1'])

    add(4445, {'height':10,'square':40,'short':10,'long':12},
        {'height':[8,9,10,12],'square':[20,30,40,50],
         'short':[8,10,12,14],'long':[16,18,20,22]},
        {'area':'height*(3*square+2*(short+long))'}, 'area',
        ['Matt paints three walls in a square room of side {square} feet and all four walls in a rectangular room measuring {short} by {long} feet. Every wall is {height} feet high. Ignoring doors and windows, how many square feet does he paint?',
         'A square living room has side length {square} feet. A bedroom is {short} feet by {long} feet. Three living-room walls and all bedroom walls, each {height} feet tall, are painted fully. What is the painted area in square feet?'],
        ['Three rectangular wall faces have length {square} feet each; two more have length {short} feet and two have length {long} feet. They are all {height} feet tall and painted without deductions. What is their total area in square feet?',
         'Matt covers three sides of a {square}-foot square room and the full perimeter of a {short}-by-{long}-foot room, from floor to a common {height}-foot height. With no openings subtracted, how many square feet of wall are covered?'],
        ['Painting three walls of a square room of side {square} feet plus all walls of a {short}-by-{long}-foot bedroom covers {area} square feet. All walls share one height and openings are ignored. What is the wall height in feet?',
         'The lengths of seven painted walls are {square}, {square}, {square}, {short}, {short}, {long}, and {long} feet. Their total face area is {area} square feet and their heights are equal. How tall are the walls in feet?'],
        'area/(3*square+2*(short+long))',
        'Use wall area, not floor area:10*(3*40+2*(10+12))=1640. Inverse divides by the sum of painted wall lengths.')

    add(3790, {'hives':5,'yield_liters':20,'jar':'0.5'},
        {'hives':[4,5,6,8],'yield_liters':[12,16,20,24],
         'jar':['0.25','0.5','0.75','1']},
        {'jars':'hives*yield_liters/(2*jar)'}, 'jars',
        ['James has {hives} hives producing {yield_liters} liters of honey each. A friend supplies jars for half the honey. Each jar holds {jar} liters. How many jars does James need for his own half, filling them exactly?',
         'Each of {hives} hives yields {yield_liters} liters of honey. James keeps half and packs it in {jar}-liter jars. The friend jars the other half separately. How many full jars does James need?'],
        ['James jars only one half of the honey from {hives} hives, each producing {yield_liters} liters. His jars hold {jar} liters each with no leftovers. What is his required jar count?',
         'Two friends split the honey from {hives} hives equally. Each hive yields {yield_liters} liters. One friend uses jars of capacity {jar} liters, all filled exactly. How many jars does that friend use?'],
        ['James keeps half the honey from {hives} hives producing {yield_liters} liters each. It exactly fills {jars} identical jars. What is the capacity of one jar in liters?',
         'Half of a harvest from {hives} hives, with {yield_liters} liters per hive, fills {jars} jars equally with no leftover honey. How many liters does each jar hold?'],
        'hives*yield_liters/(2*jars)',
        'Only half of the100-liter harvest needs James\'s jars;50/.5=100 jars. All jar counts are integer; inverse capacity may be fractional.',
        integer=False, integers=['jars'])

    add(5822, {'breakfast':73,'lunch':127,'dinner':87,'factor':2},
        {'breakfast':[40,60,80,100],'lunch':[80,100,120,140],
         'dinner':[50,70,90,110],'factor':[2,3,4,5]},
        {'friday':'breakfast+lunch+dinner','saturday':'(breakfast+lunch+dinner)*factor'}, 'saturday',
        ['A restaurant serves {breakfast} customers at breakfast, {lunch} at lunch, and {dinner} at dinner on Friday. Saturday is predicted to have {factor} times Friday\'s total. How many customer visits are predicted for Saturday?',
         'Friday customer visits number {breakfast}, {lunch}, and {dinner} over three meals. Saturday\'s forecast is {factor} times the sum. What is the forecast visit count?'],
        ['Add the restaurant\'s Friday breakfast, lunch, and dinner visits: {breakfast}, {lunch}, and {dinner}. Saturday should bring {factor} times that full-day amount. How many Saturday visits are expected?',
         'A cafe expects Saturday\'s visits to be {factor} times all Friday visits. Friday had {breakfast} breakfast visits, {lunch} lunch visits, and {dinner} dinner visits. How many visits are expected Saturday?'],
        ['Friday has {breakfast} breakfast visits, {lunch} lunch visits, and some dinner visits. A Saturday forecast of {saturday} visits equals {factor} times all Friday visits. How many Friday dinner visits were there?',
         'Saturday\'s {saturday}-visit forecast is {factor} times Friday\'s total. Friday breakfast had {breakfast} visits and lunch had {lunch}. How many visits were at Friday dinner?'],
        'saturday/factor-breakfast-lunch',
        'Multiply the sum of all three meals, not one meal:2*(73+127+87)=574. Inverse divides by the factor before subtracting known meals.')

    add(1727, {'marked':4,'cost':2,'percent':25,'flat_count':5,'flat_profit':'0.2'},
        {'marked':[2,4,6,8],'cost':[2,4,6,8],'percent':[10,20,25,50],
         'flat_count':[3,4,5,6],'flat_profit':['0.1','0.2','0.3','0.4']},
        {'profit':'marked*cost*percent/100+flat_count*flat_profit'}, 'profit',
        ['A seller sells {marked} pairs of socks at a profit of {percent} percent of their ${cost} cost per pair. Another {flat_count} pairs earn ${flat_profit} profit each. How many dollars of total profit does the seller earn?',
         'Each of {marked} sock pairs costs ${cost} and is resold with {percent} percent profit on cost. A separate {flat_count} pairs each produce ${flat_profit} profit. What is the combined profit in dollars?'],
        ['Profit has two parts: {marked} pairs each earning {percent} percent of a ${cost} purchase cost, and {flat_count} other pairs each earning ${flat_profit}. What is the total profit in dollars?',
         'A sock seller adds a {percent} percent markup on cost to {marked} pairs bought at ${cost} each, and gains a flat ${flat_profit} on each of {flat_count} different pairs. How many dollars does the seller gain overall?'],
        ['A seller earns ${profit} total profit. Of this, {marked} pairs earn {percent} percent profit on a ${cost} cost per pair. The other {flat_count} pairs each earn the same flat profit. How many dollars is that flat profit per pair?',
         'A sock seller\'s total profit is ${profit}. A first batch of {marked} pairs has ${cost} cost per pair and {percent} percent profit on cost. The remaining {flat_count} pairs each earn an equal profit. What is this per-pair profit in dollars?'],
        '(profit-marked*cost*percent/100)/flat_count',
        'The25% is profit on the$2 purchase cost, not sale-price revenue. Original4*2*.25+5*.2=3. Structural probes isolate the second batch\'s per-pair profit.',
        integer=False)

    add(5414, {'distance':'2.5','walked':1,'minutes_per_mile':20},
        {'distance':['2.5','3','3.5','4'],'walked':['0.5','1','1.5','2'],
         'minutes_per_mile':[10,15,20,25]},
        {'remaining_minutes':'(distance-walked)*minutes_per_mile'}, 'remaining_minutes',
        ['Peter must walk {distance} miles. He has already walked {walked} miles and maintains a pace of {minutes_per_mile} minutes per mile. How many more minutes does he need?',
         'A store is {distance} miles from Peter\'s starting point. After walking {walked} miles, he continues at {minutes_per_mile} minutes per mile. How many minutes remain until he reaches it?'],
        ['A walker completes {walked} miles of a {distance}-mile journey. Every mile takes {minutes_per_mile} minutes at his constant pace. How much additional time in minutes is required?',
         'Peter walks at {minutes_per_mile} minutes for each mile. His full route is {distance} miles and he has finished {walked} miles. How many minutes will the unfinished part take?'],
        ['Peter has walked {walked} miles of a journey. At {minutes_per_mile} minutes per mile, he needs {remaining_minutes} more minutes. What is the total journey distance in miles?',
         'After {walked} miles, a walker has {remaining_minutes} minutes left at a constant pace of {minutes_per_mile} minutes per mile. How long in miles is the whole route?'],
        'walked+remaining_minutes/minutes_per_mile',
        'Multiply the remaining distance, not the whole route, by minutes per mile. Original(2.5-1)*20=30. The inverse restores the already-walked distance.',
        integer=False, nonnegative=['distance-walked'])

    add(6353, {'eagle':15,'falcon':46,'pelican':33,'hummingbird':30,'time':2},
        {'eagle':[12,15,18,21],'falcon':[36,40,44,48],
         'pelican':[24,28,32,36],'hummingbird':[20,25,30,35],'time':[1,2,3,4]},
        {'total':'time*(eagle+falcon+pelican+hummingbird)'}, 'total',
        ['An eagle, falcon, pelican, and hummingbird fly at {eagle}, {falcon}, {pelican}, and {hummingbird} miles per hour respectively. Each flies for {time} hours. What is the sum of their distances in miles?',
         'Four birds maintain speeds of {eagle}, {falcon}, {pelican}, and {hummingbird} miles per hour. They all fly for {time} hours. How many miles do they fly in total across the four birds?'],
        ['For {time} hours each, four birds fly steadily at {eagle}, {falcon}, {pelican}, and {hummingbird} miles per hour. Adding their individual travel distances, how many miles do they cover?',
         'One bird travels {eagle} miles per hour, another {falcon}, a third {pelican}, and a fourth {hummingbird}. All fly for the same {time} hours. What is their combined distance in miles?'],
        ['Four birds fly for the same length of time at speeds of {eagle}, {falcon}, {pelican}, and {hummingbird} miles per hour. Their individual distances sum to {total} miles. For how many hours does each bird fly?',
         'Birds moving at {eagle}, {falcon}, {pelican}, and {hummingbird} miles per hour each fly for an equal duration. Their combined distance is {total} miles. What is that duration in hours?'],
        'total/(eagle+falcon+pelican+hummingbird)',
        'Distances from four distinct birds add; a common duration factors out. Original2*(15+46+33+30)=248. Inverse divides by the sum of rates.')

    add(5824, {'dollars':10,'found':2},
        {'dollars':[4,6,8,10,12],'found':[1,2,3,4,5]},
        {'dimes':'dollars*5+found'}, 'dimes',
        ['Barry has ${dollars} in dimes, each worth $0.10. Dan initially has half that monetary amount, all in dimes, then finds {found} more dimes. How many dimes does Dan have?',
         'One brother owns ${dollars} worth of ten-cent coins. His younger brother starts with half that value in the same coins and adds {found} found coins. How many coins does the younger brother now hold?'],
        ['Dan begins with half the dollar value of Barry\'s ${dollars} collection of dimes. Every dime is worth ten cents. Dan then picks up {found} more dimes. What is his final number of dimes?',
         'Barry\'s dimes have a total value of ${dollars}. Dan\'s dimes initially have half that value; he later finds {found} additional dimes. If a dime is $0.10, how many dimes does Dan have at the end?'],
        ['Dan initially has half the value of Barry\'s dimes, all in dimes. After finding {found} extra dimes, Dan has {dimes} dimes. Each dime is $0.10. What was the value of Barry\'s collection in dollars?',
         'The younger brother now has {dimes} dimes after finding {found}. Before that, his dimes were worth half of the older brother\'s collection. Each coin is worth ten cents. What is the older brother\'s collection worth in dollars?'],
        '(dimes-found)/5',
        'Convert dollars to dime counts using10 coins per dollar, then halve. Original10*10/2+2=52. Inverse subtracts found coins before doubling the original value.')

    add(3221, {'age':32,'offset':5,'multiple':4},
        {'age':[32,36,40,44],'offset':[2,3,4,5],'multiple':[3,4,5,6]},
        {'young':'age/2-offset','old':'(age/2-offset)*multiple'}, 'old',
        ['Kay is {age} years old. Her youngest sibling is {offset} years less than half her age. Her oldest sibling is {multiple} times as old as the youngest. How old is the oldest sibling in years?',
         'The youngest sibling of {age}-year-old Kay is {offset} years younger than half Kay\'s age. The oldest sibling\'s age is {multiple} times the youngest\'s. What is the oldest sibling\'s age in years?'],
        ['To get the youngest sibling\'s age, halve Kay\'s {age} years and subtract {offset}. The oldest is {multiple} times the youngest\'s age. How many years old is the oldest?',
         'Kay, aged {age}, has a youngest sibling whose age is half of hers minus {offset} years. Multiplying that youngest age by {multiple} gives the oldest sibling\'s age. What is that oldest age in years?'],
        ['Kay\'s youngest sibling is {offset} years less than half Kay\'s age. The oldest sibling is {multiple} times the youngest\'s age and is {old} years old. How old is Kay in years?',
         'The oldest sibling, aged {old}, is {multiple} times the age of the youngest. The youngest\'s age is {offset} years less than half Kay\'s age. What is Kay\'s age in years?'],
        '(old/multiple+offset)*2',
        'Subtract offset after halving, then multiply: (32/2-5)*4=44. Original14 siblings is irrelevant. Keep youngest below Kay and oldest at least Kay in variant domains.',
        nonnegative=['young-1','old-age'], integers=['young'])

    add(267, {'hours':7,'minutes_per_pizza':10,'flour':22,'per_pizza':'0.5'},
        {'hours':[2,3,4,5],'minutes_per_pizza':[5,10,15,20],
         'flour':[30,40,50,60],'per_pizza':['0.25','0.5','0.75','1']},
        {'sold':'hours*60/minutes_per_pizza',
         'left':'flour/per_pizza-hours*60/minutes_per_pizza'}, 'left',
        ['Jimmy continuously makes one pizza every {minutes_per_pizza} minutes for {hours} hours. He starts with {flour} kg of flour and each pizza needs {per_pizza} kg. How many more pizzas can he make from the leftover flour, using it exactly?',
         'A pizza maker works without breaks for {hours} hours at {minutes_per_pizza} minutes per pizza. A starting stock of {flour} kg flour supplies {per_pizza} kg per pizza. After this work, how many additional pizzas can the remaining flour make exactly?'],
        ['For {hours} hours, Jimmy completes a pizza every {minutes_per_pizza} minutes. All use {per_pizza} kg flour apiece from {flour} kg initially available. How many further whole pizzas use all his remaining flour?',
         'There are initially {flour} kg of flour. A baker uses {per_pizza} kg for every pizza, making pizzas continuously for {hours} hours at {minutes_per_pizza} minutes each. How many pizzas can he make afterward with the remaining flour exactly?'],
        ['Jimmy makes one pizza every {minutes_per_pizza} minutes for {hours} hours. Each needs {per_pizza} kg flour. The leftover flour makes exactly {left} additional pizzas. How many kg of flour did he have at first?',
         'A baker works for {hours} hours, completing each pizza in {minutes_per_pizza} minutes and using {per_pizza} kg flour each. Enough flour remains for {left} more pizzas. What was the initial flour stock in kg?'],
        '(hours*60/minutes_per_pizza+left)*per_pizza',
        'Continuous work gives7*60/10=42 pizzas;22/.5 total capacity leaves2. Exact whole-pizza constraints avoid floor/ceiling ambiguity. Inverse recovers flour from both batches.',
        integer=False, nonnegative=['left-1'], integers=['sold','left'])

    add(6267, {'tv_denom':5,'study_denom':4},
        {'tv_denom':[2,3,4,5,6],'study_denom':[2,3,4,5,6]},
        {'minutes':'1440*(tv_denom-1)/tv_denom/study_denom'}, 'minutes',
        ['Mia spends 1/{tv_denom} of a 24-hour day watching TV. She spends 1/{study_denom} of the remaining time studying. How many minutes per day does she study?',
         'A student watches TV for 1/{tv_denom} of an entire 24-hour day and studies for 1/{study_denom} of the time left afterward. What is the study time in minutes?'],
        ['Start with a full 24-hour day. TV takes one of {tv_denom} equal parts; studying then takes one of {study_denom} equal parts of the remainder. How many minutes are spent studying?',
         'Mia allocates 1/{tv_denom} of her day to TV, then allocates 1/{study_denom} of the unallocated time to study. A day has 24 hours. What is her daily study duration in minutes?'],
        ['After using 1/{tv_denom} of a 24-hour day for TV, Mia divides the remaining time into equal parts and studies for one part lasting {minutes} minutes. Into how many parts does she divide the remaining time?',
         'TV occupies 1/{tv_denom} of a 24-hour day. Mia studies for {minutes} minutes, which is one of some equal parts of the time left after TV. How many such equal parts make up that remaining time?'],
        '1440*(tv_denom-1)/tv_denom/minutes',
        'Second fraction applies to the remainder, not the full day:1440*(4/5)/4=288 minutes. Inverse denominator uses remaining minutes divided by study minutes.',
        integer=False)

    add(5032, {'phone':26,'tablet':53},
        {'phone':[20,24,28,32,36],'tablet':[40,45,50,55,60]},
        {'total':'tablet+phone/2'}, 'total',
        ['A charger fully charges an empty phone in {phone} minutes or an empty tablet in {tablet} minutes. Charging is linear. Ana fully charges the empty tablet, then charges the empty phone to half capacity, sequentially. How many minutes does this take?',
         'An empty phone takes {phone} minutes to charge fully and an empty tablet takes {tablet}. At constant charging rates, first charge the tablet fully and then the phone halfway. What is the total time in minutes?'],
        ['A charger handles one device at a time, charging each at a constant rate. A full tablet charge needs {tablet} minutes and a full phone charge needs {phone}. From empty, charge the tablet completely and afterward the phone to half. How many minutes are needed altogether?',
         'Ana charges devices consecutively without overlap. Empty-to-full times are {tablet} minutes for the tablet and {phone} for the phone, with linear charging. What is the time in minutes for one full tablet charge followed by half a phone charge?'],
        ['At constant charging rates, a full empty-tablet charge followed by a half empty-phone charge takes {total} minutes. The tablet alone takes {tablet} minutes. How many minutes would a full empty-phone charge take?',
         'Two sequential charges take {total} minutes: {tablet} minutes for an empty tablet to fill, then enough time to half-fill an empty phone. Charging is linear. How long would the phone need for a full charge in minutes?'],
        '(total-tablet)*2',
        'Original53+26/2=66. Probe prompts explicitly fix the original sequential/linear assumptions; half charge uses half time. Inverse doubles only the phone portion.')

    add(2503, {'old_count':50,'factor':2,'extra':60},
        {'old_count':[20,30,40,50],'factor':[2,3,4,5],'extra':[10,20,30,40]},
        {'total':'2*old_count*factor+extra'}, 'total',
        ['Alden used to have {old_count} nephews and now has {factor} times that many. Vihaan now has {extra} more nephews than Alden. How many nephews do their two counts sum to?',
         'Alden\'s current nephew count is {factor} times his earlier count of {old_count}. Vihaan has {extra} more than Alden currently does. What is the sum of their current counts?'],
        ['Multiply Alden\'s earlier {old_count} nephews by {factor} to get his current count. Vihaan\'s current count exceeds Alden\'s by {extra}. Adding the two counts, how many nephews are counted?',
         'Alden now has {factor} times the {old_count} nephews he once had; Vihaan has that current count plus {extra}. What is the combined count obtained by adding their counts?'],
        ['Alden now has {factor} times his earlier number of nephews. Vihaan has {extra} more than Alden now. Their current counts sum to {total}. What was Alden\'s earlier nephew count?',
         'The current nephew counts of Alden and Vihaan add to {total}. Vihaan\'s is {extra} larger, and Alden\'s current count is {factor} times his earlier count. How many nephews did Alden have earlier?'],
        '(total-extra)/(2*factor)',
        'Original current Alden100, Vihaan160, sum260. The old count is not the current count; the sum is2*old*factor+extra. Wording asks sum of counts, avoiding an unsupported disjoint-family assumption.')

    add(5606, {'hours':3,'vacuum':45,'dust':60,'mop':30,'per_cat':5,'cats':3},
        {'hours':[3,4,5,6],'vacuum':[20,30,40,50],'dust':[20,30,40,50],
         'mop':[10,20,30,40],'per_cat':[3,4,5,6],'cats':[2,3,4,5]},
        {'free':'60*hours-vacuum-dust-mop-per_cat*cats'}, 'free',
        ['Gunther has {hours} hours free. Vacuuming takes {vacuum} minutes, dusting {dust}, mopping {mop}, and brushing each of {cats} cats takes {per_cat} minutes. These chores are done sequentially. How many free minutes remain?',
         'A person has a {hours}-hour free period and spends {vacuum} minutes vacuuming, {dust} dusting, {mop} mopping, and {per_cat} minutes on each of {cats} cats. With no overlap, how many minutes are left?'],
        ['From {hours} hours, subtract three sequential tasks lasting {vacuum}, {dust}, and {mop} minutes plus {per_cat} minutes per cat for {cats} cats. How many unused minutes remain?',
         'Gunther cleans without overlapping tasks: {vacuum} minutes for carpets, {dust} for furniture, {mop} for floors, and {per_cat} for each of {cats} cats. Given {hours} hours available, how many minutes are unused?'],
        ['Gunther has {hours} hours, uses {vacuum} minutes vacuuming, {dust} dusting, and {mop} mopping, then brushes some cats at {per_cat} minutes per cat. He has {free} minutes left. How many cats did he brush?',
         'A {hours}-hour free period ends with {free} minutes unused after sequential tasks of {vacuum}, {dust}, and {mop} minutes and brushing cats for {per_cat} minutes each. How many cats were brushed?'],
        '(60*hours-vacuum-dust-mop-free)/per_cat',
        'Convert hours to minutes before subtracting all chores, including one brush per cat:180-45-60-30-5*3=30. Inverse divides only residual brushing time by per-cat time.',
        nonnegative=['free'])

    add(2645, {'haircuts':4,'hair_rate':30,'perms':1,'perm_rate':40,
               'dyes':2,'dye_rate':60,'dye_cost':10,'tips':50},
        {'haircuts':[2,3,4,5],'hair_rate':[20,25,30,35],
         'perms':[1,2,3,4],'perm_rate':[30,40,50,60],
         'dyes':[1,2,3,4],'dye_rate':[50,60,70,80],
         'dye_cost':[5,10,15,20],'tips':[20,30,40,50]},
        {'net':'haircuts*hair_rate+perms*perm_rate+dyes*(dye_rate-dye_cost)+tips'}, 'net',
        ['Rebecca charges ${hair_rate} per haircut, ${perm_rate} per perm, and ${dye_rate} per dye job. Each dye job costs her ${dye_cost} in dye; ignore other costs. After {haircuts} haircuts, {perms} perms, {dyes} dye jobs, and ${tips} tips, what are her net earnings in dollars?',
         'A salon performs {haircuts} haircuts at ${hair_rate}, {perms} perms at ${perm_rate}, and {dyes} dye jobs at ${dye_rate}. Dye costs ${dye_cost} per job and tips total ${tips}. Ignoring other costs, what are net earnings in dollars?'],
        ['Add receipts from {haircuts} haircuts at ${hair_rate}, {perms} perms at ${perm_rate}, and {dyes} dye services at ${dye_rate}, plus ${tips} tips. Only dye supplies cost money, at ${dye_cost} per dye service. What amount remains in dollars after supplies?',
         'A stylist earns ${hair_rate} for each of {haircuts} haircuts and ${perm_rate} for each of {perms} perms. Each of {dyes} dye jobs brings ${dye_rate} but uses ${dye_cost} supplies. With ${tips} tips and no other expenses, how many net dollars does she earn?'],
        ['A salon\'s net earnings are ${net}, including ${tips} tips. It performs {haircuts} haircuts at ${hair_rate} and {perms} perms at ${perm_rate}, plus some dye jobs bringing ${dye_rate} each and costing ${dye_cost} each. With no other costs, how many dye jobs were done?',
         'After deducting only dye costs, a stylist has ${net} including ${tips} tips, {haircuts} haircuts at ${hair_rate}, and {perms} perms at ${perm_rate}. Every dye job brings ${dye_rate} and costs ${dye_cost}. How many dye jobs account for the rest?'],
        '(net-tips-haircuts*hair_rate-perms*perm_rate)/(dye_rate-dye_cost)',
        'Dye cost is charged once per dye job, not once per day. Original120+40+2*(60-10)+50=310. Inverse uses the net per-job margin.',
        nonnegative=['dye_rate-dye_cost-1'])

    add(908, {'rate':5000,'first_hours':12,'second_hours':14,'percent':20},
        {'rate':[1000,2000,3000,4000],'first_hours':[4,6,8,10],
         'second_hours':[4,6,8,10],'percent':[10,20,25,50]},
        {'total':'rate*first_hours+rate*(100+percent)/100*second_hours'}, 'total',
        ['A telethon raises ${rate} per hour for {first_hours} hours, then raises {percent} percent more per hour for the next {second_hours} hours. How many dollars does it raise overall?',
         'John collects ${rate} hourly during a first {first_hours}-hour period. The hourly rate increases by {percent} percent for another {second_hours} hours. What is the total collection in dollars?'],
        ['For {first_hours} hours, donations arrive at ${rate} per hour. During the following {second_hours} hours the hourly rate is {percent} percent higher. How many dollars arrive across both periods?',
         'A fundraiser has two consecutive periods: {first_hours} hours at ${rate} hourly, followed by {second_hours} hours at a rate {percent} percent above the first. What is the total raised in dollars?'],
        ['A telethon raises ${total} over two periods: {first_hours} hours at an initial hourly rate and {second_hours} hours at {percent} percent more than that rate. What is the initial hourly rate in dollars?',
         'Donations total ${total}. The first {first_hours} hours share one hourly rate and the next {second_hours} hours have a rate {percent} percent higher. How many dollars per hour were collected initially?'],
        'total/(first_hours+(100+percent)/100*second_hours)',
        'Increase the second hourly rate, not the entire event total:5000*12+6000*14=144000. The inverse factors out the original rate.')

    add(4874, {'red':50,'green':40,'green_eaten':15,'multiple':2},
        {'red':[40,60,80,100],'green':[20,30,40,50],
         'green_eaten':[4,6,8,10],'multiple':[2,3,4,5]},
        {'remaining':'red+green-green_eaten*(multiple+1)'}, 'remaining',
        ['Fran bakes {red} red and {green} green macarons. She eats {green_eaten} green macarons and {multiple} times as many red ones. How many macarons remain altogether?',
         'There are {red} red cookies and {green} green cookies. A person eats {green_eaten} green cookies and {multiple} times that count of red cookies. What is the total count left?'],
        ['Of {red} red macarons and {green} green ones, Fran removes {green_eaten} green and {multiple} times that number in red. How many of both colors remain in total?',
         'Fran starts with {red} red and {green} green macarons. Her red consumption is {multiple} times her green consumption of {green_eaten}. How many macarons are uneaten?'],
        ['Fran has {red} red and {green} green macarons. After she eats {multiple} times as many red as green, {remaining} macarons remain overall. How many green macarons did she eat?',
         'Initially {red} red and {green} green cookies are available. A person eats red cookies in a count {multiple} times her green-cookie count, leaving {remaining} cookies. What is the green-cookie count eaten?'],
        '(red+green-remaining)/(multiple+1)',
        'Total consumption includes both colors:15+2*15=45 from90, leaving45. Inverse divides total eaten by multiple+1, not by multiple.',
        nonnegative=['red-green_eaten*multiple','green-green_eaten'])
    return result


def specifications():
    # Refuse to silently reinterpret the existing ten-root frozen bootstrap.
    original = read(DATA/'bootstrap_calibration_relations_v2.json')
    if bootstrap_specs() != original:
        raise ValueError('Bootstrap source no longer reproduces its frozen relations')
    authored={s['source_id']:s for s in bootstrap_specs()}
    for relation in original:
        relation['sampling_parameter_order']=list(authored[relation['source_id']]['domains'])
    return original + additional_specifications()


def main():
    sources = read(DATA/'cal40.json')
    reviews = {r['source_id']:r for r in read(REVIEWS)}
    teachers = {r['source_id']:r for r in checked_rows(OUT/'materials/teacher_targets_cal40.jsonl')}
    specs = specifications()
    by_id = {s['source_id']:s for s in specs}
    accepted = {sid for sid,r in reviews.items() if r['accepted'] is True}
    if len(by_id) != len(specs) or set(by_id) != accepted or len(sources) != 40:
        raise ValueError('Relation coverage must equal the reviewed accepted calibration cohort')
    bundles = []
    for source in sources:
        sid = source['source_id']; review = reviews[sid]; teacher = teachers[sid]
        if (review['output_sha256'] != teacher['output_sha256']
                or review['question_sha256'] != source['question_sha256']):
            raise ValueError('Teacher review does not bind the exact source/output')
        if sid not in accepted:
            continue
        relation = by_id[sid]
        bundles.append({'source_id':sid, 'spec':relation, 'variants':build(relation,source)})
    freeze(RELATIONS, [b['spec'] for b in bundles])
    freeze(VARIANTS, bundles)
    manifest = {
        'cohort':'cal40_reviewed34_v2', 'calibration_source_count':40,
        'accepted_teacher_count':len(accepted), 'roots_with_reviewed_relations':len(bundles),
        'selection_variants':sum(len(b['variants']['select']) for b in bundles),
        'audit_variants':sum(len(b['variants']['audit']) for b in bundles),
        'excluded':[{'source_id':s['source_id'],'reason':reviews[s['source_id']]['reason'],
                     'issue_tags':reviews[s['source_id']]['issue_tags']}
                    for s in sources if s['source_id'] not in accepted],
        'exclusion_scope':'development_cohort_only_not_a_frozen_final7473_policy',
        'review_kind':'assistant_mathematical_review_not_independent_human',
        'generated_questions_semantically_inspected_separately':False,
        'sampling_parameter_order_is_explicit':True,
        'original_ten_probe_instances_preserved':True,
        'supersedes_unqueued_v1_with_JSON_key_order_replay_bug':True,
        'scope_limited_to_calibration_not_full7473':True,
        'audit_role':'hyperparameter_development_not_independent_confirmation',
        'official_test_used':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in [Path(__file__),
            HERE/'variants.py', HERE/'bootstrap_relations.py', DATA/'cal40.json', REVIEWS,
            OUT/'materials/teacher_targets_cal40.jsonl', DATA/'bootstrap_calibration_relations_v2.json',
            RELATIONS, VARIANTS]}}
    freeze(MANIFEST,manifest)
    save(OUT/'preflight/cal40_reviewed_variants_v2.json',manifest)
    print(f'Reviewed calibration relations: {len(bundles)}/40 roots; '
          f'{manifest["selection_variants"]} Vsel + {manifest["audit_variants"]} Vaudit; '
          '6 quarantined; public test unused', flush=True)


if __name__ == '__main__':
    main()

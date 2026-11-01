"""Initial assistant-reviewed templates for known development roots only.

Explicitly limited coverage: these are not a general GSM8K relation parser.
No teacher output is accepted by this script, and no test root is inspected.
"""
from __future__ import annotations

from runtime_full import *
from variants import build


def templates(prefix, numeric, wording, structural, original, changed, integer=True):
    return {kind:[{'id':f'{prefix}_{kind}_{split}','question':text,'answer':answer,'integer_answer':integer}
                  for split,text in zip(['select','audit'],texts)]
            for kind,texts,answer in [('numeric',numeric,original),('wording',wording,original),('structural',structural,changed)]}


def spec(index,params,domains,derived,original,template_set,reason,constraints=()):
    return {'source_id':f'gsm8k:train:{index}','original_parameters':params,'domains':domains,
            'derived':derived,'original_answer':original,'templates':template_set,
            'nonnegative_constraints':list(constraints),
            'review':{'accepted':True,'kind':'assistant_mathematical_review','reason':reason,
                      'independent_human_review':False}}


def specifications():
    result=[]
    result.append(spec(3146,{'a':2,'b':3,'rounds':22},
        {'a':[2,3,4,5,6],'b':[3,4,5,6,7],'rounds':[8,12,16,20,24]},
        {'total':'(a+b)*rounds'},'total/(a+b)',
        templates('joint_capacity',
            ['George carries {a} buckets per round and Harry carries {b} buckets per round. They work together to fill a pool requiring {total} buckets. How many rounds are needed?',
             'A pool requires {total} buckets. One person brings {a} buckets each round and another brings {b} in the same round. How many rounds will fill it?'],
            ['Two workers complete every trip together. One delivers {a} buckets and the other delivers {b}; a pool needs {total} buckets. How many trips must they make?',
             'A pool needs {total} buckets of water. Every joint trip supplies {a} buckets from one worker and {b} from the other. The buckets are blue. How many joint trips are needed?'],
            ['George and Harry make {rounds} rounds together, carrying {a} and {b} buckets respectively in each round. How many buckets do they carry altogether?',
             'One worker delivers {a} buckets per trip and a partner delivers {b}. After {rounds} simultaneous trips, how many buckets have they delivered?'],
            'total/(a+b)','(a+b)*rounds'),
        'The simultaneous capacities add. Original110/(2+3)=22; structural variants ask the total, so multiply rather than divide.'))
    result.append(spec(6101,{'saved':30,'wage':5,'lawns':3,'weeks':6},
        {'saved':[10,20,30,40,50],'wage':[3,4,5,6,7],'lawns':[2,3,4,5],'weeks':[3,4,5,6,7]},
        {'price':'saved+wage*lawns*weeks'},'(price-saved)/(wage*lawns)',
        templates('savings',
            ['Cary wants shoes costing ${price} and has saved ${saved}. He earns ${wage} per lawn and mows {lawns} lawns each weekend. How many more weekends are needed?',
             'A pair of shoes costs ${price}. A gardener has ${saved}, earns ${wage} for each lawn, and mows {lawns} lawns every weekend. How many additional weekends will reach the price?'],
            ['A gardener starts with ${saved} toward a ${price} purchase. Every weekend brings payment for {lawns} lawns at ${wage} per lawn. How many weekends of work are needed?',
             'To afford shoes worth ${price}, a student can add earnings from {lawns} lawns each weekend at ${wage} per lawn to existing savings of ${saved}. The shoes are black. How many weekends are required?'],
            ['Cary has saved ${saved} and then mows {lawns} lawns each weekend for {weeks} weekends, earning ${wage} per lawn. How much money does he have in total?',
             'A student begins with ${saved}, earns ${wage} per lawn, and completes {lawns} lawns per weekend for {weeks} weekends. What shoe price can these total savings exactly cover?'],
            '(price-saved)/(wage*lawns)','price'),
        'Subtract existing savings, then divide by wage times lawns per weekend. Original(120-30)/(5*3)=6; inverse task asks accumulated savings.'))
    result.append(spec(575,{'servings':3,'calories':120},
        {'servings':[2,3,4,5,6],'calories':[40,60,80,100,120]},
        {'eaten':'servings*calories/2'},'eaten',
        templates('half_package',
            ['A package has {servings} servings, each containing {calories} calories. John eats half the package. How many calories does he eat?',
             'A candy packet contains {servings} servings of {calories} calories each. Half of the packet is eaten. How many calories are consumed?'],
            ['John shares a candy package equally with a friend. The package contains {servings} servings and each serving has {calories} calories. How many calories does John receive?',
             'There are {servings} equal servings in a candy bag, with {calories} calories per serving. John eats one of two equal halves. The bag is red. How many calories does he consume?'],
            ['A candy package has {servings} equal servings. Eating half the package supplies {eaten} calories. How many calories are in each serving?',
             'Half a packet supplies {eaten} calories, and the entire packet contains {servings} equal servings. What is the number of calories per serving?'],
            'eaten','eaten*2/servings'),
        'Total calories equal servings times per-serving calories; half gives180 in original. Inverse variants recover per-serving calories, not the half-package quantity.'))
    result.append(spec(5825,{'slices':8,'people':8,'meal':20},
        {'slices':[4,5,8,10],'people':[4,8,12,16],'meal':[10,20,30,40]},
        {'tomatoes':'people*meal/slices'},'tomatoes',
        templates('tomatoes',
            ['Each tomato provides {slices} slices. Each person needs {meal} slices for a meal. How many tomatoes are needed to feed {people} people?',
             'A cook needs {meal} tomato slices for each of {people} diners and cuts every tomato into {slices} slices. How many tomatoes are required?'],
            ['A family of {people} shares a meal with {meal} tomato slices per person. One tomato yields {slices} slices. How many whole tomatoes supply exactly this meal?',
             'A cook serves {people} guests. Each plate requires {meal} slices, and each green tomato yields {slices} slices. The plates are white. How many tomatoes are necessary?'],
            ['A cook slices {tomatoes} tomatoes into {slices} slices each. Each guest receives {meal} slices. How many guests can be served exactly?',
             'There are {tomatoes} tomatoes with {slices} slices each, and a meal requires {meal} slices per person. How many people can eat?'],
            'people*meal/slices','tomatoes*slices/meal'),
        'Slices needed are people times meal requirement; divide by slices per tomato. Original8*20/8=20. Structural variants change unknown to diners.'))
    result.append(spec(2727,{'large':4,'largecap':20,'smallcap':4,'small':30},
        {'large':[2,3,4,5],'largecap':[10,15,20,25],'smallcap':[2,3,4,5],'small':[10,15,20,25,30]},
        {'total':'large*largecap+small*smallcap'},'(total-large*largecap)/smallcap',
        templates('planters',
            ['Oshea has {total} seeds and {large} large planters holding {largecap} seeds each. Small planters hold {smallcap} seeds each. After filling the large planters, how many small planters are needed?',
             'There are {total} seeds. First fill {large} planters with {largecap} seeds each, then use small planters of {smallcap} seeds each. How many small planters are required?'],
            ['A gardener fills {large} large pots, each with {largecap} seeds, from a stock of {total} seeds. Every remaining pot holds {smallcap} seeds. How many remaining pots are needed?',
             'All {total} basil seeds must be planted. There are {large} large containers holding {largecap} seeds apiece and small containers holding {smallcap} apiece. The containers are green. How many small containers are needed after the large ones are filled?'],
            ['A gardener fills {large} large planters with {largecap} seeds each and {small} small planters with {smallcap} seeds each. How many seeds are planted altogether?',
             'There are {large} full large pots containing {largecap} seeds each and {small} full small pots containing {smallcap} seeds each. What is the total number of seeds?'],
            '(total-large*largecap)/smallcap','total'),
        'Remove large-planter capacity, then divide remaining seeds by small capacity. Original(200-4*20)/4=30. Structural variants ask forward total.'))
    result.append(spec(3821,{'total':32,'denom':4},
        {'total':[24,48,72,96,120],'denom':[2,3,4,6]},
        {'left':'total*(denom-1)/denom'},'left*denom/(denom-1)',
        templates('remaining_fraction',
            ['Anna spends 1/{denom} of her money and has ${left} left. How many dollars did she have originally?',
             'After spending one part out of {denom} equal parts of her money, Anna has ${left} remaining. How much money did she start with?'],
            ['Anna divides her savings into {denom} equal portions and spends one portion. The other portions total ${left}. What were her initial savings?',
             'One of {denom} equal parts of a sum is spent; the unspent amount is ${left}. Anna keeps it in a blue purse. What was the original sum?'],
            ['Anna starts with ${total} and spends 1/{denom} of it. How many dollars remain?',
             'A student has ${total} in savings and uses one of {denom} equal portions. What amount of savings is left?'],
            'left*denom/(denom-1)','left'),
        'The remaining fraction is(denom-1)/denom. Original24/(3/4)=32; structural variants supply original and ask remaining.'))
    result.append(spec(7017,{'buy':40,'sell':60,'count':20},
        {'buy':[10,20,30,40],'sell':[50,60,70,80],'count':[5,10,15,20]},
        {'profit':'(sell-buy)*count'},'profit',
        templates('rug_profit',
            ['A seller buys {count} rugs at ${buy} each and sells them at ${sell} each. What is the total profit in dollars?',
             'Each rug costs a merchant ${buy} and is sold for ${sell}. How much profit comes from selling {count} rugs?'],
            ['A merchant sells {count} identical rugs. The purchase price per rug is ${buy}, and each sale brings ${sell}. What is the total amount earned above purchase costs?',
             'A rug shop sells {count} rugs at ${sell} each after buying each for ${buy}. All rugs are blue. What is the profit in dollars?'],
            ['Each rug is bought for ${buy} and sold for ${sell}. The total profit is ${profit}. How many rugs were sold?',
             'A merchant earns ${profit} profit by reselling rugs bought for ${buy} and sold for ${sell} each. What number of rugs were sold?'],
            'profit','profit/(sell-buy)'),
        'Profit per rug is sale minus cost; original(60-40)*20=400. Structural variants solve quantity rather than profit.'))
    result.append(spec(2839,{'crackers':48,'per':2,'have':6},
        {'crackers':[40,48,56,64,72],'per':[2,4,8,10],'have':[2,4,6,8]},
        {'needed':'crackers/per-have'},'needed',
        templates('smores',
            ["Each s'more requires {per} graham crackers and one marshmallow. Jacob has {crackers} crackers and {have} marshmallows. How many more marshmallows are needed to use all the crackers?",
             'A snack uses {per} crackers and one marshmallow. There are {crackers} crackers and {have} marshmallows available. How many additional marshmallows must be bought to use all crackers?'],
            ['Jacob can make one snack from {per} crackers and one marshmallow. He wants to use all {crackers} crackers and already has {have} marshmallows. How many marshmallows must he add?',
             'A recipe pairs one marshmallow with every {per} crackers. The kitchen has {crackers} crackers and {have} marshmallows. The plate is blue. How many extra marshmallows are required to use every cracker?'],
            ['Jacob has {have} marshmallows and buys {needed} more. Each snack uses one marshmallow and {per} crackers. How many crackers are required to use all the marshmallows?',
             'A cook starts with {have} marshmallows, adds {needed}, and uses one marshmallow with {per} crackers per snack. How many crackers are needed?'],
            'needed','(have+needed)*per'),
        'Required marshmallows equal crackers/per; subtract stock. Original48/2-6=18. Inverse asks crackers given marshmallow total.',
        ['crackers/per-have']))
    result.append(spec(1203,{'first':'1.75','second':'1.25','left':6},
        {'first':['1.25','1.5','1.75','2','2.25'],'second':['0.5','0.75','1','1.25','1.5'],'left':[4,6,8,10]},
        {'total':'first+second+left'},'total',
        templates('money_before_spending',
            ['Josh spends ${first} on a drink and ${second} on another purchase. He has ${left} left. How many dollars did he have at first?',
             'A student buys a drink for ${first}, spends another ${second}, and has ${left} remaining. What was the original amount of money?'],
            ['After paying ${first} for a drink and ${second} for a snack, Josh has ${left}. What amount did he start with?',
             'Josh has ${left} after two purchases costing ${first} and ${second}. His wallet is brown. How much money did he have before either purchase?'],
            ['Josh starts with ${total}, buys a drink for ${first}, and spends ${second} more. How much money remains?',
             'A student has ${total} and pays ${first} for a drink and ${second} for a snack. What amount of money is left?'],
            'total','total-first-second',integer=False),
        'Restore both expenses to remaining money. Original1.75+1.25+6=9; structural variants subtract expenses from supplied original.'))
    result.append(spec(3339,{'original':800,'percent':20},
        {'original':[400,500,600,700,800],'percent':[10,20,30,40]},
        {'current':'original*(100+percent)/100'},'current*100/(100+percent)',
        templates('inverse_growth',
            ['A school has {current} students after its enrollment increased by {percent} percent. How many students were there before the increase?',
             'The number of students grew by {percent} percent to {current}. What was the number of students last year?'],
            ['This year a school has {current} pupils, which is {percent} percent more than last year. How many pupils attended last year?',
             'Enrollment is now {current}, following a rise of {percent} percent from the previous year. The school has a blue gate. What was the previous enrollment?'],
            ['A school starts with {original} students, then enrollment increases by {percent} percent. How many students attend afterward?',
             'There were {original} pupils last year. This year the school has {percent} percent more. What is the current number of pupils?'],
            'current*100/(100+percent)','current'),
        'An increase multiplies original by(100+percent)/100, so inverse divides by that factor. Original960/1.2=800; forward variants change unknown.'))
    for relation in result:
        if relation['source_id']=='gsm8k:train:5825':
            relation['integer_constraints']=['tomatoes']
        if relation['source_id']=='gsm8k:train:2839':
            relation['integer_constraints']=['crackers/per','needed']
    return result


def main():
    sources={r['source_id']:r for r in read(DATA/'cal40.json')}
    specs=specifications()
    result=[]
    for s in specs:
        assert s['source_id'] in sources
        built=build(s,sources[s['source_id']])
        result.append({'source_id':s['source_id'],'spec':s,'variants':built})
    freeze(DATA/'bootstrap_calibration_relations_v2.json',specs)
    freeze(DATA/'bootstrap_calibration_variants_v2.json',result)
    summary={'roots_with_reviewed_templates':len(result),'calibration_roots_total':40,
             'selection_variants':sum(len(r['variants']['select']) for r in result),
             'audit_variants':sum(len(r['variants']['audit']) for r in result),
             'coverage_is_limited_not_full_GSM8K':True,'independent_human_review':False,
             'development_audit_may_be_used_for_hyperparameter_calibration':True,
             'official_test_accessed':False}
    save(OUT/'preflight/bootstrap_variants_v2.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

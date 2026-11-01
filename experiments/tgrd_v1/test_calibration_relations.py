"""Independent quantity accounting for every added relation and inverse target."""
import unittest
from fractions import Fraction as F

from calibration_relations import additional_specifications
from runtime_full import DATA, read
from variants import build, solve


def accounting(index, raw):
    p = {k:F(str(v)) for k,v in raw.items()}
    if index == 5753:
        girls = p['total']-p['boys']; long = girls*p['long_num']/p['denom']
        return girls-long, p['total']
    if index == 4793:
        rates = [p['a'], p['a']/(1/p['factor']), p['b'], p['b']+p['extra']]
        return sum(rate*p['days'] for rate in rates), p['days']
    if index == 2386:
        stock = p['intake']/p['denom']
        for change in [p['intake'], -p['adopted'], p['kittens'], -p['returned']]:
            stock += change
        return stock, p['kittens']
    if index == 607:
        retained = p['total']/p['denom']; pool = p['total']-retained
        return retained-pool/p['siblings'], p['total']
    if index == 5574:
        daily = [p['weekday']]*5+[p['saturday'],F(0)]
        return sum(daily)*p['weeks'], p['weeks']
    if index == 1075:
        flashlight = p['hoodie']*p['flash_percent']/100
        discounted_boots = p['boots']-p['boots']*p['discount']/100
        return p['hoodie']+flashlight+discounted_boots, p['boots']
    if index == 7014:
        stock = p['initial']
        for key in ['first','floated','later']:
            stock -= p[key]
            if stock < 0: raise ValueError('Unavailable balloons removed')
        return stock+p['received'], p['received']
    if index == 4488:
        arrival_fast = p['distance']/p['fast']; arrival_slow = p['distance']/p['slow']
        return arrival_slow-arrival_fast, p['distance']
    if index == 4445:
        walls = [p['square']]*3+[p['short']]*2+[p['long']]*2
        return sum(length*p['height'] for length in walls), p['height']
    if index == 3790:
        kept_honey = sum([p['yield_liters']]*int(p['hives']))/2
        return kept_honey/p['jar'], p['jar']
    if index == 5822:
        meals = [p[k] for k in ['breakfast','lunch','dinner']]
        return sum(meals)*p['factor'], p['dinner']
    if index == 1727:
        first_cost = p['marked']*p['cost']
        first_revenue = first_cost+first_cost*p['percent']/100
        return first_revenue-first_cost+p['flat_count']*p['flat_profit'], p['flat_profit']
    if index == 5414:
        full_time = p['distance']*p['minutes_per_mile']
        spent_time = p['walked']*p['minutes_per_mile']
        return full_time-spent_time, p['distance']
    if index == 6353:
        distances = [p[k]*p['time'] for k in ['eagle','falcon','pelican','hummingbird']]
        return sum(distances), p['time']
    if index == 5824:
        dan_value = p['dollars']/2
        return dan_value/F('0.1')+p['found'], p['dollars']
    if index == 3221:
        youngest = p['age']/2-p['offset']
        return youngest*p['multiple'], p['age']
    if index == 267:
        count = p['hours']*60/p['minutes_per_pizza']
        remaining_flour = p['flour']-count*p['per_pizza']
        return remaining_flour/p['per_pizza'], p['flour']
    if index == 6267:
        day_minutes = F(24*60); tv_minutes = day_minutes/p['tv_denom']
        return (day_minutes-tv_minutes)/p['study_denom'], p['study_denom']
    if index == 5032:
        return p['tablet']+p['phone']*F(1,2), p['phone']
    if index == 2503:
        alden = p['old_count']*p['factor']; vihaan = alden+p['extra']
        return alden+vihaan, p['old_count']
    if index == 5606:
        tasks = [p['vacuum'],p['dust'],p['mop']]+[p['per_cat']]*int(p['cats'])
        return p['hours']*60-sum(tasks), p['cats']
    if index == 2645:
        receipts = sum(p[count]*p[rate] for count,rate in [
            ('haircuts','hair_rate'),('perms','perm_rate'),('dyes','dye_rate')])
        expenses = p['dyes']*p['dye_cost']
        return receipts+p['tips']-expenses, p['dyes']
    if index == 908:
        increased_rate = p['rate']+p['rate']*p['percent']/100
        return p['rate']*p['first_hours']+increased_rate*p['second_hours'], p['rate']
    if index == 4874:
        red_left = p['red']-p['green_eaten']*p['multiple']
        green_left = p['green']-p['green_eaten']
        if min(red_left,green_left) < 0: raise ValueError('Unavailable cookies eaten')
        return red_left+green_left, p['green_eaten']
    raise ValueError('Missing independent accounting oracle')


class CalibrationRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = additional_specifications()
        cls.sources = {r['source_id']:r for r in read(DATA/'cal40.json')}

    def test_originals_and_all_new_probe_labels_against_quantity_accounting(self):
        self.assertEqual(len(self.specs),24)
        for spec in self.specs:
            index = int(spec['source_id'].split(':')[-1]); source = self.sources[spec['source_id']]
            with self.subTest(root=index):
                forward,inverse = accounting(index,spec['original_parameters'])
                self.assertEqual(forward,F(source['gold']))
                for kind,expected in [('numeric',forward),('wording',forward),('structural',inverse)]:
                    for template in spec['templates'][kind]:
                        _,actual,_ = solve(spec,template,spec['original_parameters'])
                        self.assertEqual(F(actual),expected)
                bundle = build(spec,source)
                for split,rows in bundle.items():
                    self.assertEqual(len(rows),6)
                    for row in rows:
                        forward,inverse = accounting(index,row['base_parameters'])
                        self.assertEqual(F(row['gold']),inverse if row['kind']=='structural' else forward)

    def test_all_physical_counts_and_intermediate_stocks_are_valid(self):
        for spec in self.specs:
            bundle=build(spec,self.sources[spec['source_id']])
            for row in bundle['select']+bundle['audit']:
                p={k:F(str(v)) for k,v in row['base_parameters'].items()}
                index=int(spec['source_id'].split(':')[-1])
                if index==2386:
                    self.assertEqual((p['intake']/p['denom']).denominator,1)
                    self.assertGreaterEqual(p['intake']+p['intake']/p['denom']-p['adopted']-p['returned'],1)
                if index==607:
                    keep=p['total']/p['denom']; each=(p['total']-keep)/p['siblings']
                    self.assertEqual(keep.denominator,1);self.assertEqual(each.denominator,1)
                    self.assertGreater(keep,each)
                if index==267:
                    self.assertEqual((p['hours']*60/p['minutes_per_pizza']).denominator,1)
                    left,_=accounting(index,p)
                    self.assertEqual(left.denominator,1);self.assertGreater(left,0)
                if index in (7014,4874): accounting(index,p)

    def test_calibration_questions_have_no_duplicate_or_cross_split_instance(self):
        all_questions=set()
        for spec in self.specs:
            bundle=build(spec,self.sources[spec['source_id']]);seen=set()
            for row in bundle['select']+bundle['audit']:
                key=tuple(sorted(row['base_parameters'].items()))
                self.assertNotIn(key,seen);seen.add(key)
                self.assertNotIn(row['question'],all_questions);all_questions.add(row['question'])
        self.assertEqual(len(all_questions),24*12)


if __name__=='__main__': unittest.main()

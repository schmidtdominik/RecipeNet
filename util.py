import numpy as np
import math

class Recipes:

    def __init__(self):
        self.load_recipes()

    def load_recipes(self):
        with np.load('simplified-recipes-1M.npz', allow_pickle=True) as data:
            self.recipes = data['recipes']
            self.ingredients = data['ingredients']
        np.random.shuffle(self.recipes)

        self.train_size = int(0.97 * len(self.recipes))
        self.test_size = len(self.recipes) - self.train_size
        self.train_recipes = self.recipes[:self.train_size]
        self.test_recipes = self.recipes[self.train_size:]
        print(self.train_recipes.shape, self.test_recipes.shape)

    def onehot_to_hr_recipe(self, ind):
        r = []
        for i, v in enumerate(ind):
            if v == 1:
                r.append(self.ingredients[i])
        return len(r), r

    """def get_batch_alt(recipes_array, batch_size, set_n_to_zero=1):
        while True:
            print(len(recipes_array), 'dataset iterator epoch start')
            offset = 0
            for batch_end_offset in range(batch_size, len(recipes_array)+1, batch_size):

                batch_data = recipes_array[batch_end_offset-batch_size:batch_end_offset]

                batch_y = np.zeros((batch_size, len(ingredients)))
                for i, row in enumerate(batch_data):
                    try:
                        batch_y[i, row] = 1
                    except IndexError: pass

                batch_x = np.copy(batch_y)
                for i, row in enumerate(batch_data):
                    try:
                        ind = np.random.choice(row, set_n_to_zero, replace=False)
                        batch_x[i, ind] = 0
                    except (IndexError, ValueError): pass

                #indices = np.random.random((batch_size, len(ingredients))) > 0.1
                #batch_x[indices] = 0
                yield batch_x, batch_y"""

    def labels_to_batch(self, l):
        l = self.filter_ingr(l)
        assert len(l) > 2
        k = [self.hr_to_indices_list(l)]

        x, y = self.batch_iterator(k, 1).__next__()
        return x + y

    def hr_to_indices_list(self, l):
        r = np.zeros(len(l), dtype=np.int32)
        for c, i in enumerate(l):
            r[c] = np.where(self.ingredients == i)[0][0]
        return r

    def batch_iterator(self, recipes_array, batch_size):
        while True:
            # print('\n', len(recipes_array), 'dataset iterator epoch start')
            offset = 0
            for batch_end_offset in range(batch_size, len(recipes_array) + 1, batch_size):
                batch_data = recipes_array[batch_end_offset - batch_size:batch_end_offset]

                batch_y = np.zeros((batch_size, len(self.ingredients)))
                batch_x = np.zeros((batch_size, len(self.ingredients)))

                for i, row in enumerate(batch_data):
                    remove_max_n_elements = 5
                    remove_max_one_nth_of_elements = 4
                    try:
                        n_to_remove = np.random.randint(1, 1 + max(0, min(remove_max_n_elements,
                                                                          math.floor(len(row) / remove_max_one_nth_of_elements))))

                        ind = np.random.randint(0, len(row), n_to_remove)
                        row_with_ingr_removed = np.delete(row, ind)
                        batch_y[i, row[ind]] = 1
                        batch_x[i, row_with_ingr_removed] = 1
                    except (IndexError, ValueError):
                        pass

                yield batch_x, batch_y

    def get_batch_count(self, recipes_array, bs):
        return len(list(range(bs, len(recipes_array), bs)))

    def filter_ingr(self, l):
        return [k for k in l if k in self.ingredients or print(k, 'not found')]

    def match_ingr(self, i):
        l = []
        for u in self.ingredients:
            if i in u or u in i:
                l.append(u)
        return l

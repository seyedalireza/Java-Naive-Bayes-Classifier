package de.daslaboratorium.machinelearning.classifier.bayes;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collection;
import java.util.Comparator;
import java.util.SortedSet;
import java.util.TreeSet;

import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.Classifier;

/**
 * A concrete implementation of the abstract Classifier class.  The Bayes
 * classifier implements a naive Bayes approach to classifying a given set of
 * features: classify(feat1,...,featN) = argmax(P(cat)*PROD(P(featI|cat)
 *
 * @param <T> The feature class.
 * @param <K> The category class.
 * @author Philipp Nolte
 * <p>
 * // * @see http://en.wikipedia.org/wiki/Naive_Bayes_classifier
 */
public class BayesClassifier<T, K> extends Classifier<T, K> {

    /**
     * Calculates the product of all feature probabilities: PROD(P(featI|cat)
     *
     * @param features The set of features to use.
     * @param category The category to test for.
     * @return The product of all feature probabilities.
     */
    private BigDecimal featuresProbabilityProduct(Collection<T> features,
                                                  K category) {
        BigDecimal product = new BigDecimal(1.0);
        for (T feature : features)
            product = product.multiply(this.featureWeighedAverage(feature, category));
        return product;
    }

    /**
     * Calculates the probability that the features can be classified as the
     * category given.
     *
     * @param features The set of features to use.
     * @param category The category to test for.
     * @return The probability that the features can be classified as the
     * category.
     */
    private BigDecimal categoryProbability(Collection<T> features, K category) {
        return new BigDecimal(this.getCategoryCount(category))
                .divide(new BigDecimal(this.getCategoriesTotal()), 2, RoundingMode.HALF_UP)
                .multiply(featuresProbabilityProduct(features, category));
    }

    /**
     * Retrieves a sorted <code>Set</code> of probabilities that the given set
     * of features is classified as the available categories.
     *
     * @param features The set of features to use.
     * @return A sorted <code>Set</code> of category-probability-entries.
     */
    private SortedSet<Classification<T, K>> categoryProbabilities(
            Collection<T> features) {

        /*
         * Sort the set according to the possibilities. Because we have to sort
         * by the mapped value and not by the mapped key, we can not use a
         * sorted tree (TreeMap) and we have to use a set-entry approach to
         * achieve the desired functionality. A custom comparator is therefore
         * needed.
         */
        SortedSet<Classification<T, K>> probabilities =
                new TreeSet<Classification<T, K>>(
                        new Comparator<Classification<T, K>>() {

                            public int compare(Classification<T, K> o1,
                                               Classification<T, K> o2) {

                                int toReturn = o1.getProbability().compareTo(o2.getProbability());
                                if ((toReturn == 0)
                                        && !o1.getCategory().equals(o2.getCategory()))
                                    toReturn = -1;
                                return toReturn;
                            }
                        });

        for (K category : this.getCategories())
            probabilities.add(new Classification<T, K>(
                    features, category,
                    this.categoryProbability(features, category)));
        return probabilities;
    }

    /**
     * Classifies the given set of features.
     *
     * @return The category the set of features is classified as.
     */
    @Override
    public Classification<T, K> classify(Collection<T> features) {
        SortedSet<Classification<T, K>> probabilites =
                this.categoryProbabilities(features);

        if (!probabilites.isEmpty()) {
            return probabilites.last();
        }
        return null;
    }

    /**
     * Classifies the given set of features. and return the full details of the
     * classification.
     *
     * @return The set of categories the set of features is classified as.
     */
    public Collection<Classification<T, K>> classifyDetailed(
            Collection<T> features) {
        return this.categoryProbabilities(features);
    }

}

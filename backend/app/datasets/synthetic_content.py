"""
Synthetic dataset content, structured as topic -> list of (title, passage_text).

Kept as data, not embedded in chunking/retrieval/app logic (Section 6).
Passages are written to be genuinely distinguishable topically and lexically,
so hybrid retrieval, chunking, and reranking evaluations later actually mean
something instead of operating on lorem-ipsum noise.

Each topic has multiple passages of varying length (some short, some long
enough to require chunking) so the chunking strategies have real material
to differ on.
"""

TOPICS: dict[str, list[tuple[str, str]]] = {
    "machine_learning": [
        (
            "Introduction to Supervised Learning",
            "Supervised learning is a machine learning paradigm where a model is trained on "
            "labeled data. Each training example consists of an input and a corresponding "
            "correct output. The goal is for the model to learn a mapping function that can "
            "predict outputs for new, unseen inputs. Common supervised learning tasks include "
            "classification, where the output is a discrete category, and regression, where "
            "the output is a continuous value. Popular algorithms include linear regression, "
            "logistic regression, decision trees, random forests, and support vector machines. "
            "The quality of a supervised model depends heavily on the quality and quantity of "
            "labeled training data available.",
        ),
        (
            "Overfitting and Regularization",
            "Overfitting occurs when a machine learning model learns the training data too "
            "well, including its noise and outliers, resulting in poor generalization to new "
            "data. A model that overfits will show high accuracy on training data but low "
            "accuracy on test data. Regularization techniques such as L1 and L2 penalties, "
            "dropout in neural networks, and early stopping are commonly used to prevent "
            "overfitting. Cross-validation is another important technique used to detect "
            "overfitting by evaluating model performance on held-out data during training.",
        ),
        (
            "Neural Network Architectures",
            "Neural networks are composed of layers of interconnected nodes, or neurons, "
            "inspired loosely by biological brains. A basic feedforward neural network "
            "passes data through an input layer, one or more hidden layers, and an output "
            "layer. Convolutional neural networks, or CNNs, are specialized for processing "
            "grid-like data such as images, using convolutional filters to detect spatial "
            "patterns. Recurrent neural networks, or RNNs, and their variant LSTMs are "
            "designed for sequential data such as text or time series. Transformer "
            "architectures, introduced in 2017, use self-attention mechanisms and have "
            "become the dominant architecture for natural language processing tasks.",
        ),
        (
            "Gradient Descent Optimization",
            "Gradient descent is an iterative optimization algorithm used to minimize a loss "
            "function by updating model parameters in the direction of steepest descent. "
            "The learning rate controls the size of each update step. Stochastic gradient "
            "descent, or SGD, updates parameters using a single training example or small "
            "batch at a time, which is more computationally efficient than using the entire "
            "dataset. Variants such as Adam, RMSprop, and momentum-based methods adapt the "
            "learning rate during training to improve convergence speed and stability.",
        ),
    ],
    "indian_history": [
        (
            "The Indus Valley Civilization",
            "The Indus Valley Civilization, also known as the Harappan Civilization, "
            "flourished around 2500 BCE in the northwestern regions of South Asia, in "
            "present-day Pakistan and northwest India. Major urban centers included "
            "Mohenjo-daro and Harappa, known for their advanced urban planning, standardized "
            "brick sizes, sophisticated drainage systems, and grid-pattern streets. The "
            "civilization had a writing system that remains undeciphered to this day. Trade "
            "networks extended to Mesopotamia, and archaeological evidence suggests a "
            "relatively egalitarian society without obvious monumental palaces or temples.",
        ),
        (
            "The Mauryan Empire",
            "The Mauryan Empire was founded by Chandragupta Maurya around 322 BCE and became "
            "one of the largest empires in ancient India, eventually covering most of the "
            "Indian subcontinent. His grandson, Ashoka the Great, is one of the most famous "
            "rulers of the empire. After the bloody conquest of Kalinga, Ashoka embraced "
            "Buddhism and promoted non-violence, tolerance, and welfare through edicts carved "
            "on pillars and rocks across the empire. The Mauryan administration was highly "
            "centralized, with an extensive bureaucracy and spy network described in the "
            "Arthashastra, a treatise attributed to Chanakya.",
        ),
        (
            "The Mughal Empire",
            "The Mughal Empire was established in 1526 by Babur following his victory at the "
            "First Battle of Panipat. The empire reached its zenith under Akbar, Jahangir, "
            "Shah Jahan, and Aurangzeb. Akbar is particularly noted for his policy of "
            "religious tolerance and administrative reforms including the mansabdari system. "
            "Shah Jahan commissioned the Taj Mahal as a mausoleum for his wife Mumtaz Mahal, "
            "regarded as one of the finest examples of Mughal architecture. The empire "
            "gradually declined through the eighteenth century due to succession conflicts, "
            "regional rebellions, and the growing influence of the British East India Company.",
        ),
        (
            "The Indian Independence Movement",
            "The Indian independence movement was a series of historic events aimed at ending "
            "British colonial rule in India, culminating in independence on August 15, 1947. "
            "Mahatma Gandhi led numerous campaigns based on nonviolent civil disobedience, "
            "including the Non-Cooperation Movement, the Salt March of 1930, and the Quit "
            "India Movement of 1942. Other key figures included Jawaharlal Nehru, Subhas "
            "Chandra Bose, and Sardar Vallabhbhai Patel. The movement drew participation from "
            "across religious and regional lines, though it was ultimately accompanied by the "
            "partition of India and Pakistan, which caused significant communal violence and "
            "mass displacement.",
        ),
    ],
    "climate_science": [
        (
            "The Greenhouse Effect",
            "The greenhouse effect is a natural process where certain gases in Earth's "
            "atmosphere trap heat from the sun, keeping the planet warm enough to sustain "
            "life. Greenhouse gases including carbon dioxide, methane, and water vapor allow "
            "sunlight to pass through the atmosphere but absorb and re-emit infrared radiation "
            "reflected from Earth's surface. Human activities, particularly the burning of "
            "fossil fuels, deforestation, and industrial processes, have significantly "
            "increased atmospheric concentrations of these gases since the Industrial "
            "Revolution, enhancing the natural greenhouse effect and driving global warming.",
        ),
        (
            "Ocean Acidification",
            "Ocean acidification refers to the ongoing decrease in the pH of Earth's oceans, "
            "caused primarily by the uptake of carbon dioxide from the atmosphere. As oceans "
            "absorb roughly a quarter of human-emitted carbon dioxide, the seawater becomes "
            "more acidic, which can interfere with the ability of marine organisms such as "
            "corals, mollusks, and some plankton species to form calcium carbonate shells and "
            "skeletons. This process threatens marine ecosystems, coral reefs, and fisheries "
            "that hundreds of millions of people depend on for food and livelihoods.",
        ),
        (
            "Renewable Energy Transition",
            "The transition to renewable energy involves shifting from fossil fuel-based "
            "power generation to sources such as solar, wind, hydroelectric, and geothermal "
            "energy. Solar photovoltaic and wind turbine costs have fallen dramatically over "
            "the past decade, making renewables cost-competitive with fossil fuels in many "
            "regions. Challenges to widespread adoption include the intermittent nature of "
            "solar and wind generation, the need for improved energy storage technologies "
            "such as batteries, and the modernization of electrical grid infrastructure to "
            "handle distributed and variable power sources.",
        ),
    ],
    "human_biology": [
        (
            "The Human Circulatory System",
            "The circulatory system, also called the cardiovascular system, consists of the "
            "heart, blood vessels, and blood, and is responsible for transporting oxygen, "
            "nutrients, hormones, and waste products throughout the body. The heart is a "
            "muscular organ that pumps blood through two main circuits: the pulmonary "
            "circuit, which carries blood to and from the lungs for oxygenation, and the "
            "systemic circuit, which delivers oxygenated blood to the rest of the body. "
            "Arteries carry blood away from the heart, while veins return blood to the heart, "
            "and capillaries facilitate the exchange of gases and nutrients at the tissue level.",
        ),
        (
            "The Immune System",
            "The immune system is a complex network of cells, tissues, and organs that "
            "defends the body against pathogens such as bacteria, viruses, fungi, and "
            "parasites. The innate immune system provides a rapid, non-specific first line "
            "of defense, including physical barriers like skin and mucous membranes, as well "
            "as white blood cells such as neutrophils and macrophages. The adaptive immune "
            "system develops a targeted response to specific pathogens through B cells, which "
            "produce antibodies, and T cells, which can directly kill infected cells. Memory "
            "cells formed during an infection allow for a faster response upon re-exposure to "
            "the same pathogen, which is the basis for how vaccines work.",
        ),
        (
            "Sleep and the Circadian Rhythm",
            "The circadian rhythm is an internal biological clock that regulates roughly "
            "24-hour cycles of physiological processes, including the sleep-wake cycle. It is "
            "primarily controlled by the suprachiasmatic nucleus in the hypothalamus, which "
            "responds to light exposure detected by the eyes. Sleep occurs in cycles of "
            "roughly 90 minutes, alternating between non-REM stages, which are important for "
            "physical restoration, and REM sleep, which is associated with dreaming and "
            "memory consolidation. Chronic sleep disruption has been linked to impaired "
            "cognitive function, weakened immune response, and increased risk of "
            "cardiovascular disease and metabolic disorders.",
        ),
    ],
    "software_engineering": [
        (
            "Version Control with Git",
            "Git is a distributed version control system that tracks changes to source code "
            "over time, allowing multiple developers to collaborate on a project. Unlike "
            "centralized version control systems, every Git user has a full copy of the "
            "repository history on their local machine. Core concepts include commits, which "
            "are snapshots of the codebase, branches, which allow parallel lines of "
            "development, and merges, which combine changes from different branches. Common "
            "workflows involve creating feature branches, opening pull requests for code "
            "review, and merging changes into a main branch once approved.",
        ),
        (
            "RESTful API Design",
            "REST, or Representational State Transfer, is an architectural style for "
            "designing networked applications, commonly used for web APIs. RESTful APIs "
            "organize functionality around resources, identified by URLs, and use standard "
            "HTTP methods such as GET, POST, PUT, and DELETE to perform operations on those "
            "resources. Good REST API design emphasizes statelessness, meaning each request "
            "contains all information needed to process it, consistent and predictable URL "
            "structures, appropriate use of HTTP status codes, and versioning strategies to "
            "avoid breaking existing clients when the API evolves.",
        ),
        (
            "Database Indexing",
            "A database index is a data structure that improves the speed of data retrieval "
            "operations at the cost of additional storage space and slower write operations. "
            "The most common type is the B-tree index, which maintains sorted data and allows "
            "for efficient searching, insertion, and deletion in logarithmic time. Indexes "
            "are typically created on columns that are frequently used in WHERE clauses, JOIN "
            "conditions, or ORDER BY statements. Over-indexing can degrade write performance "
            "and increase storage costs, so index design requires balancing query performance "
            "against write throughput and storage constraints.",
        ),
    ],
}

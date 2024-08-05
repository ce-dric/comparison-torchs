#include <torch/torch.h>
#include <iostream>
#include <chrono>


struct VGG19 : torch::nn::Module {
    torch::nn::Sequential features;
    torch::nn::Sequential classifier;

    VGG19(int num_classes = 1000) {
        // Features part
        features = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)), torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Classifier part
        classifier = torch::nn::Sequential(
            torch::nn::Linear(512 * 7 * 7, 4096), torch::nn::ReLU(), torch::nn::Dropout(),
            torch::nn::Linear(4096, 4096), torch::nn::ReLU(), torch::nn::Dropout(),
            torch::nn::Linear(4096, num_classes)
        );

        // Register modules
        register_module("features", features);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);
        x = x.view({ x.size(0), -1 });  // Flatten the output
        x = classifier->forward(x);
        return x;
    }
};

int main() {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_train;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_train;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_val;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_val;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_total;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_total;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_program;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_program;

    std::chrono::duration<double> total_train_time(0);
    std::chrono::duration<double> total_valid_time(0);

    start_program = std::chrono::high_resolution_clock::now();
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    }
    else {
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        return 0;
    }

    int batch_size = 64;
    int epochs = 100;
    torch::Tensor input;
    torch::Tensor labels;
    torch::Tensor val_input;
    torch::Tensor val_labels;
    torch::Tensor prediction;
    torch::Tensor loss;
    torch::Tensor val_prediction;
    torch::Tensor val_loss;

    auto model = std::make_shared<VGG19>(10);  // 모델의 출력을 10개 클래스로 설정
    model->to(torch::kCUDA);  // GPU로 모델을 옮기기 (GPU가 없다면 torch::kCPU)

    // 손실 함수와 옵티마이저
    //torch::nn::CrossEntropyLoss criterion;
    torch::nn::L1Loss criterion;
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    // 학습 루프
    start_total = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        start_train = std::chrono::high_resolution_clock::now();
        model->train();
        float train_loss = 0.0;
        for (int iter = 0; iter < 10; iter++) {
            input = torch::rand({ batch_size,3,224,224 }).to(torch::kCUDA);
            labels = torch::rand({ batch_size,10 }).to(torch::kCUDA);

            prediction = model->forward(input);
            loss = criterion(prediction, labels);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_loss += loss.item<float>();
        }

        std::cout << "Epoch " << epoch << ": Average Training Loss = " << (train_loss / 10) << std::endl;
        torch::cuda::synchronize();
        end_train = std::chrono::high_resolution_clock::now();
        total_train_time += (end_train - start_train);


        // 검증 단계
        start_val = std::chrono::high_resolution_clock::now();
        model->eval();  // 모델을 평가 모드로 전환
        float total_loss = 0.0;
        torch::NoGradGuard no_grad;  // 기울기 계산 비활성화
        for (int iter = 0; iter < 10; iter++) {
            val_input = torch::rand({ batch_size,3,224,224 }).to(torch::kCUDA);
            val_labels = torch::rand({ batch_size,10 }).to(torch::kCUDA);
            val_prediction = model->forward(val_input);
            val_loss = criterion(val_prediction, val_labels);
            total_loss += val_loss.item<float>();
        }
        std::cout << "Epoch " << epoch << ": Average Validation Loss = " << (total_loss / 10) << std::endl;
        torch::cuda::synchronize();
        end_val = std::chrono::high_resolution_clock::now();
        total_valid_time += (end_val - start_val);
    }
    torch::cuda::synchronize();
    end_total = std::chrono::high_resolution_clock::now();
    end_program = std::chrono::high_resolution_clock::now();

    std::cout << "Results" << std::endl;

    std::chrono::duration<double> train_time = end_train - start_train;
    std::cout << "Average Train Time per Epoch: " << (total_train_time.count() / epochs) << " seconds." << std::endl;

    std::chrono::duration<double> valid_time = end_val - start_val;
    std::cout << "Average Valid Time per Epoch: " << (total_valid_time.count() / epochs) << " seconds." << std::endl;

    std::cout << "Total Train Time: " << total_train_time.count() << " seconds." << std::endl;

    std::cout << "Total Valid Time: " << total_valid_time.count() << " seconds." << std::endl;

    std::chrono::duration<double> total_time = end_total - start_total;
    std::cout << "Total Time: " << total_time.count() << " seconds." << std::endl;

    std::chrono::duration<double> total_program_time = end_program - start_program;
    std::cout << "Total Program Time: " << total_program_time.count() << " seconds." << std::endl;

    return 0;
}

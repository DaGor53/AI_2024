import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import train_test_split
import time
import zipfile
import gdown
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor

start_time = time.time()  # Засекаем время начала

print('Этап 0/5. Проверка и загрузка недостающих данных')

# Указываем путь к папке
folder_path = r"CGHD1152\CGHD1152"
folder_path2 = r"CGHD1152.zip"

# Проверяем, существует ли папка
if os.path.isdir(folder_path):
    print(f"Папка '{folder_path}' существует.")
    
elif os.path.exists(folder_path2):
    print('Идёт распаковка файлов')
    
    zip_path = "CGHD1152.zip"
    extract_to = "CGHD1152"
    # Открываем архив
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Распаковываем все файлы
        zip_ref.extractall(extract_to)
        print(f"Архив успешно распакован в папку: {extract_to}")
    

else:
    print(f"Папка '{folder_path}' не существует.")
    print('Производится закачка файла')
    url = "https://drive.google.com/uc?id=1kdXtkL_UiH-tzvcxHJJLZfeL9DTVOuc8"
    output = "CGHD1152.zip"
    gdown.download(url, output, quiet=False, verify=False)
    print("Файл успешно загружен.")


    print('Идёт распаковка файлов')
    zip_path = "CGHD1152.zip"
    extract_to = "CGHD1152"
    # Открываем архив
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Распаковываем все файлы
        zip_ref.extractall(extract_to)
        print(f"Архив успешно распакован в папку: {extract_to}")
    

print('Этап 1/5 Загрузка файлов из папки')

# Получение всех файлов
img_dir = r"CGHD1152\CGHD1152"
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Разделение на тренировочный и тестовый наборы
train_files, test_files = train_test_split(img_files, test_size=0.2, random_state=42)

# Функция для получения количества классов
def get_num_classes(xml_dir):
    class_names = set()
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            name_element = root.find('object/name')
            if name_element is not None:
                class_names.add(name_element.text)
            else:
                print(f"Warning: 'object/name' not found in {xml_file}")
    return len(class_names), sorted(class_names)

# Получаем количество классов и их имена
xml_dir = img_dir  # Путь к XML файлам
num_classes, class_ids = get_num_classes(xml_dir)
print(f"Количество классов: {num_classes}")
print(f"Идентификаторы классов: {class_ids}")


# Пример словаря с начальными значениями
class_count = {
    'capacitor.polarized': 0,
    'capacitor.unpolarized': 0, 
    'crossover': 0, 
    'diac': 0, 
    'diode': 0, 
    'diode.light_emitting': 0, 
    'fuse': 0, 
    'gnd': 0, 
    'inductor': 0, 
    'integrated_circuit': 0, 
    'integrated_cricuit.ne555': 0, 
    'junction': 0, 
    'lamp': 0, 
    'motor': 0, 
    'not': 0, 
    'operational_amplifier': 0, 
    'optocoupler': 0, 
    'probe.current': 0, 
    'relay': 0, 
    'resistor': 0, 
    'resistor.adjustable': 0, 
    'schmitt_trigger': 0, 
    'socket': 0, 
    'speaker': 0, 
    'switch': 0, 
    'terminal': 0, 
    'text': 0, 
    'thyristor': 0, 
    'transformer': 0, 
    'transistor': 0, 
    'transistor.photo': 0, 
    'varistor': 0, 
    'voltage.dc': 0, 
    'voltage.dc_ac': 0, 
    'vss': 0, 
    'xor': 0
}

# Функция для подсчета количества объектов по классам
def count_classes_in_dataset(xml_dir, files):
    for img_file in files:
        xml_file = img_file.replace('.jpg', '.xml')  
        xml_path = os.path.join(xml_dir, xml_file)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for object_element in root.findall('object'):
                name_element = object_element.find('name')
                if name_element is not None:
                    class_name = name_element.text
                    if class_name in class_count:
                        class_count[class_name] += 1
                else:
                    print(f"Warning: 'object/name' not found in {xml_file}")

count_classes_in_dataset(img_dir, img_files)

print("Количество объектов в датасете по классам:")
for class_name, count in class_count.items():
    print(f"{class_name}: {count}")
    

total_count = sum(class_count.values())
print(f"Общее количество объектов: {total_count}")

print('Этап 2/5 Создание класса CustomDataset')

# Класс Dataset для загрузки данных
class CustomDataset(Dataset):
    def __init__(self, img_dir, xml_dir, img_files, transform=None):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.img_files = img_files
        self.transform = transform

    # Пример маппинга строковых меток в числовые
        self.class_map = {
            'capacitor.polarized': 0,
            'capacitor.unpolarized': 1,
            'crossover': 2,
            'diac' : 3,
            'diode' : 4,
            'diode.light_emitting' : 5,
            'fuse' : 6,
            'gnd' : 7,
            'inductor' : 8,
            'integrated_circuit' : 9,
            'integrated_cricuit.ne555' : 10,
            'junction' : 11,
            'lamp' : 12,
            'motor' : 13,
            'not' : 14,
            'operational_amplifier' : 15,
            'optocoupler' : 16,
            'probe.current' : 17,
            'relay' : 18,
            'resistor' : 19,
            'resistor.adjustable' : 20,
            'schmitt_trigger' : 21,
            'socket' : 22,
            'speaker' : 23,
            'switch' : 24,
            'terminal' : 25,
            'text' : 26,
            'thyristor' : 27,
            'transformer' : 28,
            'transistor' : 29,
            'transistor.photo' : 30,
            'varistor' : 31,
            'voltage.dc' : 32,
            'voltage.dc_ac' : 33,
            'vss' : 34,
            'xor' : 35
        }
    

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        xml_path = os.path.join(self.xml_dir, self.img_files[idx].replace('.jpg', '.xml'))

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Извлечение строки с именем класса
        class_name = root.find('object/name').text

        # Преобразование строки в числовой идентификатор
        label = self.class_map.get(class_name, -1)  # если класс не найден, возвращаем -1

        if self.transform:
            image = self.transform(image)

        return image, label

print('Этап 3/5 Подготовка к обучению')

# Подготовка трансформаций
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Создание тренировочного и тестового датасета
train_dataset = CustomDataset(img_dir=img_dir, xml_dir=img_dir, img_files=train_files, transform=transform)
test_dataset = CustomDataset(img_dir=img_dir, xml_dir=img_dir, img_files=test_files, transform=transform)

# Даталоадеры
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Предобученная модель ResNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Удаляем последний слой (fc) для извлечения признаков
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Функция для извлечения признаков с использованием ResNet
def extract_features(dataloader, model, device):
    model.eval()  # Устанавливаем модель в режим оценки
    features = []
    labels = []
    with torch.no_grad():  # Отключаем вычисление градиентов
        for images, label_batch in dataloader:
            images = images.to(device)
            outputs = model(images)  # Получаем признаки
            features.append(outputs.cpu().numpy())  # Сохраняем признаки
            labels.extend(label_batch.numpy())  # Сохраняем метки
    return np.vstack(features), np.array(labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Переместим feature_extractor на устройство
feature_extractor.to(device)

# Извлекаем признаки с обучающего набора
train_features, train_labels = extract_features(train_loader, feature_extractor, device)

# Применяем PCA для понижения размерности
n_components = 512  # Количество компонент
pca = PCA(n_components=n_components)
train_features_pca = pca.fit_transform(train_features.reshape(train_features.shape[0], -1))

print(f"Размерность после PCA: {train_features_pca.shape}")
# Визуализация
plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], alpha = 0.7)
plt.title('PCA Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Преобразуем пониженные признаки в формат, подходящий для полносвязной сети
train_features_pca_tensor = torch.tensor(train_features_pca, dtype=torch.float32).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

# Создаем новую модель для классификации с пониженными признаками
model_fc = nn.Sequential(
    nn.Linear(pca.n_components_, 128),  # Полносвязный слой
    nn.ReLU(),
    nn.Linear(128, num_classes)  # Выходной слой
)

# Перемещаем модель на нужное устройство
model_fc.to(device)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model_fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print('Этап 4/5 Обучение модели')

# Обучение модели
num_epochs = 120
train_losses = []
for epoch in range(num_epochs):
    model_fc.train()
    optimizer.zero_grad()

    # Подадим пониженные признаки в модель
    outputs = model_fc(train_features_pca_tensor)  # Подали признаки в модель
    loss = criterion(outputs, train_labels_tensor)
    
    loss.backward()
    optimizer.step()
    train_losses.append(loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Сохранение модели
torch.save(model.state_dict(), 'model.pth')
print("Модель сохранена в 'model.pth'")

print('Этап 5/5 Проверка модели на тестовом датасете')

# Исправление модели для тестирования
model_fc.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Извлекаем признаки с помощью feature_extractor
        features = feature_extractor(images)
        features = features.view(features.size(0), -1)  # Преобразуем в одномерный вектор
        
        # Применяем PCA для понижения размерности
        features_pca = pca.transform(features.cpu().numpy())  # Применяем PCA только к признакам
        features_pca_tensor = torch.tensor(features_pca, dtype=torch.float32).to(device)
        
        # Передаем признаки в модель fc
        outputs = model_fc(features_pca_tensor)
 
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Точность модели: {accuracy:.2f}%')

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, num_epochs + 1)),  # Список от 1 до num_epochs
    y=[loss.item() for loss in train_losses],  # Преобразуем объекты потерь в числа
    mode='lines',
    name='Train Loss'))


fig.update_layout(
    title='Loss',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    template='plotly_dark'
)
fig.show()

end_time = time.time()  

execution_time = end_time - start_time  
print(f"Время выполнения: {execution_time:.5f} секунд")
